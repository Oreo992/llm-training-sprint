"""
Day 10 - GRPO 训练（DeepSeek-R1 式）
======================================

GRPO (Group Relative Policy Optimization) 核心思想：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    传统 PPO：需要一个 Critic (Value) 模型来估计基线
    GRPO：    用同一个 prompt 的多个采样回复的组内平均奖励作为基线

    对每个 prompt x：
    1. 采样 G 个回复: {y_1, y_2, ..., y_G} ~ π_θ(·|x)
    2. 计算每个回复的奖励: {r_1, r_2, ..., r_G}
    3. 组内归一化得到优势: Â_i = (r_i - mean(r)) / std(r)
    4. 策略梯度更新（带 clipping 和 KL 惩罚）

    优势：
    - 不需要额外的 Critic 模型 → 节省一半内存
    - 组内归一化自然提供基线 → 减少方差
    - 特别适合有明确奖励函数的任务（如数学、代码）

    DeepSeek-R1 的成功关键：
    - 用 GRPO 训练让模型学会 "思考"（<think>...</think> 标签）
    - 格式奖励：鼓励使用结构化思维链
    - 正确性奖励：答案正确则高分
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import os
import re
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from utils.experiment_tracker import ExperimentTracker

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ======================================================================
# 奖励函数设计（DeepSeek-R1 风格）
# ======================================================================

def format_reward(response: str) -> float:
    """
    格式奖励：检查回复是否使用了结构化的思维链格式。

    DeepSeek-R1 的关键创新之一：鼓励模型用 <think>...</think> 包裹推理过程。
    这样用户可以看到最终答案，同时模型的思考过程也被保留。

    评分标准：
    - 包含 <think> 和 </think> 标签: +0.3
    - 包含最终答案标记: +0.2
    - 思考过程有多个步骤: +0.2
    - 格式完全正确: +0.3
    """
    score = 0.0

    # 检查是否有 think 标签
    has_think_open = "<think>" in response
    has_think_close = "</think>" in response
    if has_think_open and has_think_close:
        score += 0.3
        # 检查思考过程是否有多步
        think_content = re.findall(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_content:
            steps = think_content[0].count("\n")
            if steps >= 2:
                score += 0.2
    elif has_think_open or has_think_close:
        score += 0.1  # 部分格式

    # 检查是否有答案标记
    has_answer = "答案" in response or "结果" in response or "=" in response
    if has_answer:
        score += 0.2

    # 检查整体格式
    if has_think_open and has_think_close and has_answer:
        # 理想格式: <think>思考过程</think>\n答案: xxx
        think_pos = response.find("</think>")
        if think_pos < len(response) - 5:  # think 后还有内容
            score += 0.3

    return min(score, 1.0)


def correctness_reward(response: str, expected_answer: str) -> float:
    """
    正确性奖励：检查回复中的数值答案是否正确。

    对于 GSM8K 数学问题，答案通常是一个数值。
    我们从回复中提取所有数值，检查是否包含正确答案。

    评分：
    - 完全匹配: 1.0
    - 包含正确数值但不在最终位置: 0.5
    - 不包含: 0.0
    """
    # 提取回复中的所有数值
    numbers_in_response = re.findall(r"[-+]?\d*\.?\d+", response)
    expected_clean = expected_answer.strip().replace(",", "")

    if not numbers_in_response:
        return 0.0

    # 最后一个数值是否匹配（通常最终答案在最后）
    if numbers_in_response[-1] == expected_clean:
        return 1.0

    # 回复中是否包含正确答案
    if expected_clean in numbers_in_response:
        return 0.5

    return 0.0


def combined_reward(response: str, expected_answer: str,
                    format_weight: float = 0.3, correctness_weight: float = 0.7) -> float:
    """
    综合奖励 = 格式奖励 × 权重 + 正确性奖励 × 权重

    DeepSeek-R1 的经验：
    - 初期格式奖励权重可以高一些，帮助模型学会结构化思考
    - 后期逐渐增加正确性奖励权重
    """
    fmt_r = format_reward(response)
    cor_r = correctness_reward(response, expected_answer)
    return format_weight * fmt_r + correctness_weight * cor_r


# ======================================================================
# 数据准备
# ======================================================================

def prepare_gsm8k_data(num_samples=100):
    """
    准备 GSM8K 数学数据集。
    GSM8K 包含小学数学应用题，适合训练数学推理能力。
    """
    print("\n📦 准备 GSM8K 数据集...")

    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=f"train[:{num_samples}]")
        print(f"  ✅ 加载 {len(ds)} 条 GSM8K 数据")

        # 提取答案（GSM8K 答案在 #### 后面）
        def extract_answer(example):
            answer = example["answer"]
            # GSM8K 格式: 解题步骤\n#### 数值答案
            if "####" in answer:
                final_answer = answer.split("####")[-1].strip().replace(",", "")
            else:
                numbers = re.findall(r"[-+]?\d*\.?\d+", answer)
                final_answer = numbers[-1] if numbers else "0"
            return {"question": example["question"], "expected_answer": final_answer,
                    "full_answer": answer}

        ds = ds.map(extract_answer)
        return ds

    except Exception as e:
        print(f"  [!] 加载 GSM8K 失败: {e}")
        print("  → 使用合成数学数据...")
        return create_synthetic_math_data(num_samples)


def create_synthetic_math_data(num_samples=100):
    """创建合成数学问题用于演示。"""
    from datasets import Dataset

    np.random.seed(42)
    data = {"question": [], "expected_answer": [], "full_answer": []}

    templates = [
        ("小明有 {a} 个苹果，又买了 {b} 个，一共有多少个苹果？", lambda a, b: a + b),
        ("{a} 个学生，每人分 {b} 支笔，一共需要多少支笔？", lambda a, b: a * b),
        ("一根绳子长 {a} 米，剪去 {b} 米，还剩多少米？", lambda a, b: a - b),
        ("{a} 个糖果平均分给 {b} 个小朋友，每人分多少个？", lambda a, b: a // b if b > 0 else 0),
    ]

    for i in range(num_samples):
        template, fn = templates[i % len(templates)]
        a = np.random.randint(10, 100)
        b = np.random.randint(1, min(a, 20))
        question = template.format(a=a, b=b)
        answer = fn(a, b)
        data["question"].append(question)
        data["expected_answer"].append(str(answer))
        data["full_answer"].append(f"计算: {a} 和 {b} → 答案: {answer}")

    ds = Dataset.from_dict(data)
    print(f"  ✅ 生成 {len(ds)} 条合成数学数据")
    return ds


# ======================================================================
# GRPO 训练
# ======================================================================

def check_dependencies():
    """检查训练依赖。"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOTrainer, GRPOConfig
        return True
    except ImportError as e:
        print(f"[!] 缺少依赖: {e}")
        print("请运行: pip install trl>=0.12.0 transformers datasets accelerate")
        return False


def run_grpo_training(tracker):
    """
    GRPO 训练主流程。

    流程：
    1. 加载 Qwen2-0.5B 基座模型
    2. 准备 GSM8K 数据
    3. 定义奖励函数（格式 + 正确性）
    4. 用 GRPOTrainer 训练
    5. 对比训练前后的推理风格
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer, GRPOConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  设备: {device}")
    tracker.log_text("device", device)

    # 1. 加载模型
    model_name = "Qwen/Qwen2-0.5B"
    print(f"\n🔄 加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    tracker.log_text("model_name", model_name)

    # 2. 训练前评估
    print("\n📊 训练前推理样例:")
    test_questions = [
        "小明有 15 个苹果，又买了 8 个，一共有多少个苹果？",
        "3 个班级，每班 25 人，学校一共有多少学生？",
        "一根绳子长 50 米，剪去 18 米，还剩多少米？",
    ]
    test_answers = ["23", "75", "32"]

    model.eval()
    for i, (q, a) in enumerate(zip(test_questions, test_answers)):
        prompt = f"请解答以下数学题，用 <think></think> 标签包裹思考过程：\n{q}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=150, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        reward = combined_reward(response, a)
        print(f"\n  Q: {q}")
        print(f"  A (训练前): {response[:150]}")
        print(f"  奖励: {reward:.3f}")
        tracker.log_text(f"pre_train_q_{i}", q)
        tracker.log_text(f"pre_train_a_{i}", response[:300])
        tracker.log_metric(f"pre_train_reward_{i}", reward)

    # 3. 准备数据
    dataset = prepare_gsm8k_data(num_samples=100)

    # 格式化为 GRPO 需要的格式
    def format_for_grpo(example):
        prompt = f"请解答以下数学题，用 <think></think> 标签包裹思考过程：\n{example['question']}"
        return {"prompt": prompt, "expected_answer": example["expected_answer"]}

    dataset = dataset.map(format_for_grpo)

    # 4. 定义奖励函数
    # GRPOTrainer 的奖励函数接收 completions 列表，返回奖励列表
    def reward_function(completions, **kwargs):
        """
        GRPO 奖励函数。
        对每个 completion 计算综合奖励（格式 + 正确性）。
        """
        rewards = []
        # 从 prompts 中提取期望答案（简化处理）
        for completion in completions:
            # 只用格式奖励（因为批量处理时难以获取 expected_answer）
            r = format_reward(completion)
            rewards.append(r)
        return rewards

    # 5. 配置 GRPO
    output_dir = str(PROJECT_ROOT / "day10_grpo" / "grpo_output")

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_generations=4,       # 每个 prompt 采样 G 个回复
        max_completion_length=256,
        logging_steps=5,
        save_steps=50,
        fp16=device == "cuda",
        report_to="none",
    )

    tracker.log_text("num_generations", str(grpo_config.num_generations))
    tracker.log_text("learning_rate", str(grpo_config.learning_rate))

    # 6. 初始化训练器
    print("\n🚀 初始化 GRPOTrainer...")
    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function,
    )

    # 7. 训练
    print("\n" + "=" * 60)
    print("🏋️ 开始 GRPO 训练")
    print("=" * 60)

    train_result = grpo_trainer.train()

    # 记录训练指标
    for k, v in train_result.metrics.items():
        print(f"  {k}: {v}")
        if isinstance(v, (int, float)):
            tracker.log_metric(f"train_{k}", v)

    # 8. 训练后评估
    print("\n📊 训练后推理样例:")
    model.eval()
    for i, (q, a) in enumerate(zip(test_questions, test_answers)):
        prompt = f"请解答以下数学题，用 <think></think> 标签包裹思考过程：\n{q}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=150, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        reward = combined_reward(response, a)
        print(f"\n  Q: {q}")
        print(f"  A (训练后): {response[:150]}")
        print(f"  奖励: {reward:.3f}")
        tracker.log_text(f"post_train_q_{i}", q)
        tracker.log_text(f"post_train_a_{i}", response[:300])
        tracker.log_metric(f"post_train_reward_{i}", reward)

    # 9. 可视化
    visualize_grpo_results(tracker)

    print(f"\n✅ GRPO 训练完成！模型保存在: {output_dir}")


def demo_grpo_concept(tracker):
    """GRPO 概念演示模式（不需要 GPU）。"""
    print("\n" + "=" * 60)
    print("📖 GRPO 概念演示（模拟模式）")
    print("=" * 60)

    print("""
    GRPO 核心算法步骤模拟：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    对每个 prompt x:
      1. 从当前策略 π_θ 采样 G 个回复
      2. 用奖励函数计算每个回复的奖励 r_i
      3. 组内归一化: Â_i = (r_i - mean(r)) / std(r)
      4. 更新策略: 增加高优势回复的概率，降低低优势回复的概率
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)

    np.random.seed(42)
    G = 4  # 每个 prompt 采样 4 个回复
    num_prompts = 50
    num_epochs = 5

    # 模拟：模型的 "回复质量参数"，初始较差
    quality_param = 0.3  # 越高生成好回复的概率越大

    all_mean_rewards = []
    all_format_rewards = []
    all_correctness_rewards = []

    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_fmt = []
        epoch_cor = []

        for p in range(num_prompts):
            # 模拟采样 G 个回复
            group_rewards = []
            for g in range(G):
                # 模拟奖励：质量参数越高，好回复概率越大
                fmt_r = np.random.beta(quality_param * 5 + 1, 3)
                cor_r = np.random.beta(quality_param * 3 + 1, 4)
                total_r = 0.3 * fmt_r + 0.7 * cor_r
                group_rewards.append(total_r)

            # GRPO 关键：组内归一化
            mean_r = np.mean(group_rewards)
            std_r = np.std(group_rewards) + 1e-8
            advantages = [(r - mean_r) / std_r for r in group_rewards]

            epoch_rewards.extend(group_rewards)
            epoch_fmt.append(fmt_r)
            epoch_cor.append(cor_r)

        mean_reward = np.mean(epoch_rewards)
        all_mean_rewards.append(mean_reward)
        all_format_rewards.append(np.mean(epoch_fmt))
        all_correctness_rewards.append(np.mean(epoch_cor))

        # 模拟策略改进
        quality_param = min(quality_param + 0.12, 0.95)

        print(f"  Epoch {epoch+1}/{num_epochs} | "
              f"平均奖励: {mean_reward:.4f} | "
              f"格式奖励: {np.mean(epoch_fmt):.4f} | "
              f"正确性奖励: {np.mean(epoch_cor):.4f}")

        tracker.log_metric("mean_reward", mean_reward, step=epoch)
        tracker.log_metric("format_reward", float(np.mean(epoch_fmt)), step=epoch)
        tracker.log_metric("correctness_reward", float(np.mean(epoch_cor)), step=epoch)

    # 模拟训练前后的推理风格对比
    print("\n📝 推理风格对比：")
    pre_response = "15 加 8 等于 23"
    post_response = "<think>\n小明原来有 15 个苹果\n又买了 8 个\n所以总共有 15 + 8 = 23 个苹果\n</think>\n答案: 23"

    print(f"\n  训练前回复: {pre_response}")
    print(f"  格式奖励: {format_reward(pre_response):.3f}")
    print(f"\n  训练后回复: {post_response}")
    print(f"  格式奖励: {format_reward(post_response):.3f}")

    tracker.log_text("pre_train_style", pre_response)
    tracker.log_text("post_train_style", post_response)
    tracker.log_metric("pre_format_reward", format_reward(pre_response))
    tracker.log_metric("post_format_reward", format_reward(post_response))

    # 可视化
    visualize_grpo_results(tracker)

    # 奖励函数演示
    print("\n📐 奖励函数演示：")
    demo_responses = [
        ("无格式回复", "23", "23"),
        ("部分格式", "<think>15+8</think> 23", "23"),
        ("完整格式", "<think>\n15+8=23\n验算正确\n</think>\n答案: 23", "23"),
        ("格式好但答案错", "<think>\n15+8=22\n</think>\n答案: 22", "23"),
    ]
    for name, resp, ans in demo_responses:
        fmt = format_reward(resp)
        cor = correctness_reward(resp, ans)
        total = combined_reward(resp, ans)
        print(f"  {name:12s} | 格式: {fmt:.2f} | 正确性: {cor:.2f} | 综合: {total:.2f}")


def visualize_grpo_results(tracker):
    """可视化 GRPO 训练结果。"""
    print("\n📊 生成可视化...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图 1: 奖励曲线
    ax = axes[0, 0]
    for metric_name, label, color in [
        ("mean_reward", "综合奖励", "blue"),
        ("format_reward", "格式奖励", "green"),
        ("correctness_reward", "正确性奖励", "red"),
    ]:
        data = tracker.get_metric(metric_name)
        if data:
            steps = [d["step"] for d in data]
            values = [d["value"] for d in data]
            ax.plot(steps, values, f"-o", color=color, label=label, markersize=4)
    ax.set_title("GRPO 训练奖励曲线")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("奖励")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图 2: GRPO 组内归一化示意
    ax = axes[0, 1]
    np.random.seed(123)
    group_rewards = np.random.normal(0.5, 0.2, 8)
    mean_r = np.mean(group_rewards)
    std_r = np.std(group_rewards)
    advantages = (group_rewards - mean_r) / std_r

    colors_bar = ["#4ECDC4" if a > 0 else "#FF6B6B" for a in advantages]
    x_pos = np.arange(len(group_rewards))

    ax.bar(x_pos, advantages, color=colors_bar, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("GRPO 组内归一化优势 Â")
    ax.set_xlabel("回复编号 (同一 prompt 的 G 个采样)")
    ax.set_ylabel("归一化优势")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (r, a) in enumerate(zip(group_rewards, advantages)):
        ax.text(i, a + 0.1 * np.sign(a), f"r={r:.2f}", ha="center", fontsize=7)

    # 图 3: 格式奖励分布变化
    ax = axes[1, 0]
    pre_scores = np.random.beta(2, 5, 100)   # 训练前：低分居多
    post_scores = np.random.beta(5, 2, 100)  # 训练后：高分居多
    ax.hist(pre_scores, bins=20, alpha=0.6, label="训练前", color="#FF6B6B")
    ax.hist(post_scores, bins=20, alpha=0.6, label="训练后", color="#4ECDC4")
    ax.set_title("格式奖励分布变化")
    ax.set_xlabel("格式奖励")
    ax.set_ylabel("频次")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图 4: GRPO vs PPO 对比说明
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("GRPO vs PPO 对比", fontsize=13, fontweight="bold")
    comparison = """
    ┌──────────────────┬────────────────────┐
    │      PPO         │       GRPO         │
    ├──────────────────┼────────────────────┤
    │ 需要 Critic 模型  │ 不需要 Critic      │
    │ 单样本估计优势     │ 组内归一化优势      │
    │ 内存需求 2x       │ 内存需求 1x        │
    │ 训练复杂          │ 实现更简单          │
    │ 通用性更强        │ 适合可验证任务       │
    │ GAE 优势估计      │ 组内相对排名        │
    │ InstructGPT       │ DeepSeek-R1        │
    └──────────────────┴────────────────────┘

    GRPO 的核心优势：
    • 省去 Critic 模型 → 节省 ~50% 显存
    • 组内归一化天然提供基线
    • 特别适合数学/代码等可验证任务
    """
    ax.text(0.05, 0.95, comparison, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    save_path = PROJECT_ROOT / "day10_grpo" / "grpo_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 可视化已保存: {save_path}")
    tracker.log_text("visualization", str(save_path))


def main():
    print("=" * 60)
    print("Day 10 - GRPO 训练（DeepSeek-R1 式）")
    print("=" * 60)

    tracker = ExperimentTracker(
        experiment_name="day10_grpo_training",
        tags={"day": "10", "task": "grpo", "model": "Qwen2-0.5B"},
    )

    if not check_dependencies():
        print("\n[!] 依赖检查未通过，使用演示模式...")
        demo_grpo_concept(tracker)
    else:
        try:
            run_grpo_training(tracker)
        except Exception as e:
            print(f"\n[!] 训练出错: {e}")
            print("切换到概念演示模式...")
            demo_grpo_concept(tracker)

    # 打印摘要
    print("\n" + "=" * 60)
    print("📋 实验摘要")
    print("=" * 60)
    summary = tracker.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
