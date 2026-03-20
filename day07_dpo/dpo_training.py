"""
Day 07 - DPO (Direct Preference Optimization) 实战训练
=====================================================

核心数学直觉：
    DPO 的损失函数本质上在做一件事：
    - 让模型对 "chosen"（人类偏好的回复）的生成概率 **提高**
    - 让模型对 "rejected"（人类不喜好的回复）的生成概率 **降低**
    - 同时用 reference model 做约束，防止模型偏离太远（KL 散度约束）

    数学公式：
        L_DPO(π_θ; π_ref) = -E[ log σ( β · (log π_θ(y_w|x)/π_ref(y_w|x)
                                              - log π_θ(y_l|x)/π_ref(y_l|x)) ) ]
    其中：
        y_w = chosen (winner) 回复
        y_l = rejected (loser) 回复
        β = 温度参数，控制偏离 reference model 的程度
        σ = sigmoid 函数

    直觉理解：
        - log π_θ(y_w|x)/π_ref(y_w|x) 是 chosen 回复在当前模型和参考模型之间的对数概率比
        - 训练目标是拉大 chosen 和 rejected 之间的"隐式奖励"差距
        - β 越大，模型越保守（越接近参考模型）
        - 与 RLHF 不同，DPO 不需要单独训练奖励模型，直接从偏好数据学习
"""

import sys
import os
import json
from pathlib import Path

# === 路径设置：确保能 import utils.experiment_tracker ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from utils.experiment_tracker import ExperimentTracker

# 中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def check_dependencies():
    """检查所需依赖是否安装。"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOTrainer, DPOConfig
        from peft import LoraConfig
        from datasets import load_dataset
        return True
    except ImportError as e:
        print(f"[!] 缺少依赖: {e}")
        print("请运行: pip install trl peft transformers datasets accelerate")
        return False


def prepare_preference_data(tokenizer, num_samples=200):
    """
    准备偏好数据集。
    尝试加载 argilla/ultrafeedback-binarized-preferences，
    如果下载失败则构造合成偏好数据用于演示。

    DPO 数据格式要求每条样本包含：
    - prompt: 用户输入
    - chosen: 人类偏好的回复
    - rejected: 人类不偏好的回复
    """
    print("\n📦 准备偏好数据集...")

    try:
        from datasets import load_dataset
        print("  尝试加载 argilla/ultrafeedback-binarized-preferences ...")
        ds = load_dataset(
            "argilla/ultrafeedback-binarized-preferences-cleaned",
            split=f"train[:{num_samples}]",
        )
        # 转换为 DPO 格式
        def format_example(example):
            prompt = example.get("instruction", example.get("prompt", ""))
            chosen = example.get("chosen", "")
            rejected = example.get("rejected", "")
            # 如果 chosen/rejected 是列表（多轮对话），取最后一条
            if isinstance(chosen, list):
                chosen = chosen[-1].get("content", str(chosen[-1]))
            if isinstance(rejected, list):
                rejected = rejected[-1].get("content", str(rejected[-1]))
            return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

        ds = ds.map(format_example)
        print(f"  ✅ 成功加载 {len(ds)} 条偏好数据")
        return ds

    except Exception as e:
        print(f"  [!] 加载在线数据集失败: {e}")
        print("  → 使用合成偏好数据...")
        return create_synthetic_preference_data(num_samples)


def create_synthetic_preference_data(num_samples=200):
    """
    构造合成偏好数据用于演示。
    chosen 回复更详细、更有帮助；rejected 回复更简短、不够好。
    """
    from datasets import Dataset

    # 定义一些 prompt-chosen-rejected 三元组模板
    templates = [
        {
            "prompt": "请解释什么是机器学习？",
            "chosen": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中自动学习和改进，而无需显式编程。主要分为监督学习、无监督学习和强化学习三大类。监督学习使用标注数据来训练模型预测结果；无监督学习从未标注数据中发现隐藏模式；强化学习通过与环境交互获得奖励来学习最优策略。",
            "rejected": "机器学习就是让机器学东西。",
        },
        {
            "prompt": "Python 和 Java 有什么区别？",
            "chosen": "Python 和 Java 的主要区别包括：1) 类型系统：Python 是动态类型，Java 是静态类型；2) 语法：Python 语法简洁，使用缩进表示代码块，Java 使用大括号；3) 执行方式：Python 是解释执行，Java 编译为字节码在 JVM 上运行；4) 应用场景：Python 常用于数据科学和脚本，Java 常用于企业应用和 Android 开发。",
            "rejected": "一个用缩进一个用大括号。",
        },
        {
            "prompt": "如何提高编程能力？",
            "chosen": "提高编程能力的有效方法：1) 坚持每日编程练习，可以使用 LeetCode 等平台；2) 阅读优秀开源项目的源代码，学习设计模式；3) 参与开源项目贡献，获得代码审查反馈；4) 系统学习数据结构与算法；5) 构建个人项目，解决实际问题；6) 写技术博客总结所学，教是最好的学。",
            "rejected": "多写代码就行了。",
        },
        {
            "prompt": "什么是深度学习？",
            "chosen": "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的层次化表示。其核心思想是通过堆叠多个非线性变换层，让模型自动学习从原始输入到目标输出的复杂映射。关键组件包括：卷积层（处理图像）、循环层（处理序列）、注意力机制（Transformer 架构）等。深度学习在计算机视觉、自然语言处理等领域取得了突破性成果。",
            "rejected": "就是很多层的神经网络。",
        },
        {
            "prompt": "请给我一个排序算法的例子。",
            "chosen": "这里是快速排序的 Python 实现：\n\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n\n快速排序的平均时间复杂度为 O(n log n)，空间复杂度为 O(log n)。它使用分治策略，选择一个基准元素，将数组分为小于和大于基准的两部分，递归排序。",
            "rejected": "用 sort() 就行。",
        },
    ]

    data = {"prompt": [], "chosen": [], "rejected": []}
    for i in range(num_samples):
        t = templates[i % len(templates)]
        # 添加微小变化避免完全重复
        suffix = f"（示例 {i+1}）" if i >= len(templates) else ""
        data["prompt"].append(t["prompt"] + suffix)
        data["chosen"].append(t["chosen"])
        data["rejected"].append(t["rejected"])

    ds = Dataset.from_dict(data)
    print(f"  ✅ 生成 {len(ds)} 条合成偏好数据")
    return ds


def run_sft_baseline(model, tokenizer, tracker, device):
    """
    SFT 基线：记录 SFT 模型的回复质量（训练前 / DPO 训练前的基线）。
    """
    print("\n" + "=" * 60)
    print("📊 SFT 基线评估（DPO 训练前）")
    print("=" * 60)

    test_prompts = [
        "请解释什么是强化学习？",
        "如何学好数学？",
        "Python 的装饰器是什么？",
    ]

    model.eval()
    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n  Prompt: {prompt}")
        print(f"  SFT 回复: {response[:200]}")
        tracker.log_text(f"sft_baseline_prompt_{i}", prompt)
        tracker.log_text(f"sft_baseline_response_{i}", response[:500])

    return test_prompts


def run_dpo_training(tracker):
    """
    DPO 训练主流程：
    1. 加载 SFT 预训练模型
    2. 配置 LoRA（低秩适应）减少训练参数
    3. 用 DPOTrainer 在偏好数据上训练
    4. 对比 SFT-only vs SFT+DPO 的回复质量
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOTrainer, DPOConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  设备: {device}")
    tracker.log_text("device", device)

    # ------------------------------------------------------------------
    # 1. 加载基础模型（作为 SFT 基线）
    # ------------------------------------------------------------------
    # 使用较小的模型以便在有限资源上训练
    model_name = "Qwen/Qwen2-0.5B"
    print(f"\n🔄 加载模型: {model_name}")
    tracker.log_text("model_name", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型（DPOTrainer 需要 model 和 ref_model）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    # ------------------------------------------------------------------
    # 2. SFT 基线评估
    # ------------------------------------------------------------------
    test_prompts = run_sft_baseline(model, tokenizer, tracker, device)

    # ------------------------------------------------------------------
    # 3. 配置 LoRA
    # ------------------------------------------------------------------
    # LoRA 通过低秩分解大幅减少可训练参数
    # r=8 表示低秩矩阵的秩，越大表达能力越强但参数越多
    # lora_alpha=16 是缩放因子，通常设为 r 的 2 倍
    # target_modules 指定对哪些层应用 LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📐 模型总参数: {total_params:,}")
    tracker.log_metric("total_params", total_params)

    # ------------------------------------------------------------------
    # 4. 准备偏好数据
    # ------------------------------------------------------------------
    train_dataset = prepare_preference_data(tokenizer, num_samples=200)

    # ------------------------------------------------------------------
    # 5. 配置 DPO 训练
    # ------------------------------------------------------------------
    output_dir = str(PROJECT_ROOT / "day07_dpo" / "dpo_output")

    # DPO 超参数说明：
    # beta: DPO 损失中的温度参数 β
    #   - β 越大 → 策略越接近参考模型（保守）
    #   - β 越小 → 策略更自由地偏离参考模型（激进）
    #   - 常用值: 0.1 ~ 0.5
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        beta=0.1,  # DPO 温度参数
        logging_steps=5,
        save_steps=50,
        max_length=512,        # prompt + response 的最大总长度
        max_prompt_length=256, # prompt 的最大长度
        remove_unused_columns=False,
        fp16=device == "cuda",
        report_to="none",  # 不使用 wandb 等外部跟踪
    )

    tracker.log_text("dpo_beta", str(dpo_config.beta))
    tracker.log_text("learning_rate", str(dpo_config.learning_rate))

    # ------------------------------------------------------------------
    # 6. 初始化 DPOTrainer
    # ------------------------------------------------------------------
    # DPOTrainer 内部会：
    # - 自动创建 reference model（冻结的原始模型副本）
    # - 计算 chosen 和 rejected 的对数概率
    # - 按 DPO 损失公式反向传播
    print("\n🚀 初始化 DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 记录可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA 可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    tracker.log_metric("trainable_params", trainable_params)
    tracker.log_metric("trainable_ratio_pct", 100 * trainable_params / total_params)

    # ------------------------------------------------------------------
    # 7. 开始 DPO 训练
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🏋️ 开始 DPO 训练")
    print("=" * 60)

    train_result = dpo_trainer.train()

    # 记录训练指标
    train_metrics = train_result.metrics
    print("\n📊 训练指标:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v}")
        if isinstance(v, (int, float)):
            tracker.log_metric(f"train_{k}", v)

    # ------------------------------------------------------------------
    # 8. DPO 训练后评估 & 对比
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 SFT+DPO 模型评估（训练后）")
    print("=" * 60)

    model.eval()
    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n  Prompt: {prompt}")
        print(f"  DPO 回复: {response[:200]}")
        tracker.log_text(f"dpo_response_{i}", response[:500])

    # ------------------------------------------------------------------
    # 9. 可视化对比
    # ------------------------------------------------------------------
    visualize_results(tracker)

    print("\n✅ DPO 训练完成！")
    print(f"  模型保存在: {output_dir}")
    print(f"  实验记录: {tracker.experiment_dir}")


def visualize_results(tracker):
    """可视化训练结果和 SFT vs DPO 对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图 1: DPO 损失曲线
    loss_data = tracker.get_metric("train_train_loss")
    if loss_data:
        steps = [d["step"] for d in loss_data]
        values = [d["value"] for d in loss_data]
        axes[0].plot(steps, values, "b-o", markersize=3)
        axes[0].set_title("DPO 训练损失")
        axes[0].set_xlabel("步数")
        axes[0].set_ylabel("损失")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "暂无损失数据", ha="center", va="center", fontsize=14)
        axes[0].set_title("DPO 训练损失")

    # 图 2: SFT vs DPO 回复质量对比（文字展示）
    axes[1].axis("off")
    axes[1].set_title("SFT vs DPO 回复质量对比", fontsize=13)

    comparison_text = "SFT-only vs SFT+DPO 对比\n" + "=" * 40 + "\n\n"
    for i in range(3):
        sft_records = tracker.texts.get(f"sft_baseline_response_{i}", [])
        dpo_records = tracker.texts.get(f"dpo_response_{i}", [])
        prompt_records = tracker.texts.get(f"sft_baseline_prompt_{i}", [])

        prompt = prompt_records[-1]["text"][:30] if prompt_records else "N/A"
        sft_resp = sft_records[-1]["text"][:50] if sft_records else "N/A"
        dpo_resp = dpo_records[-1]["text"][:50] if dpo_records else "N/A"

        comparison_text += f"Q: {prompt}...\n"
        comparison_text += f"  SFT: {sft_resp}...\n"
        comparison_text += f"  DPO: {dpo_resp}...\n\n"

    axes[1].text(0.05, 0.95, comparison_text, transform=axes[1].transAxes,
                 fontsize=8, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    save_path = PROJECT_ROOT / "day07_dpo" / "dpo_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📈 可视化结果已保存: {save_path}")


def main():
    print("=" * 60)
    print("Day 07 - DPO (Direct Preference Optimization) 实战训练")
    print("=" * 60)

    # 初始化实验跟踪器
    tracker = ExperimentTracker(
        experiment_name="day07_dpo_training",
        tags={"day": "07", "task": "dpo", "method": "DPO+LoRA"},
    )

    if not check_dependencies():
        print("\n[!] 依赖检查未通过，使用演示模式...")
        demo_dpo_concept(tracker)
        return

    try:
        run_dpo_training(tracker)
    except Exception as e:
        print(f"\n[!] 训练出错: {e}")
        print("切换到概念演示模式...")
        demo_dpo_concept(tracker)

    # 打印实验摘要
    print("\n" + "=" * 60)
    print("📋 实验摘要")
    print("=" * 60)
    summary = tracker.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")


def demo_dpo_concept(tracker):
    """
    DPO 概念演示模式（不需要 GPU）。
    用简单数值模拟 DPO 损失计算过程。
    """
    import numpy as np

    print("\n" + "=" * 60)
    print("📖 DPO 概念演示（模拟模式）")
    print("=" * 60)

    print("""
    DPO 核心思想：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    传统 RLHF 流程：
      1. SFT → 2. 训练奖励模型 → 3. PPO 强化学习

    DPO 简化流程：
      1. SFT → 2. 直接从偏好数据优化（跳过奖励模型和 PPO！）

    关键洞察：
      最优的 RLHF 策略可以用一个封闭式公式表示：
      π*(y|x) = π_ref(y|x) · exp(r(x,y) / β) / Z(x)

      反过来，奖励函数可以从策略中恢复：
      r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)

      把这个代入 Bradley-Terry 偏好模型，就得到 DPO 损失！
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)

    # 模拟 DPO 训练过程
    np.random.seed(42)
    beta = 0.1
    num_steps = 100

    # 模拟：chosen 和 rejected 的对数概率比
    # 训练过程中，chosen 的概率比应该升高，rejected 的应该降低
    log_ratio_chosen = np.zeros(num_steps)
    log_ratio_rejected = np.zeros(num_steps)
    losses = np.zeros(num_steps)

    for step in range(num_steps):
        # 模拟训练：chosen 概率逐渐提高
        log_ratio_chosen[step] = 0.5 * np.log(step + 1) + np.random.randn() * 0.1
        # 模拟训练：rejected 概率逐渐降低
        log_ratio_rejected[step] = -0.3 * np.log(step + 1) + np.random.randn() * 0.1

        # DPO 损失: -log σ(β * (log_ratio_chosen - log_ratio_rejected))
        diff = beta * (log_ratio_chosen[step] - log_ratio_rejected[step])
        losses[step] = -np.log(1 / (1 + np.exp(-diff)) + 1e-10)

        if step % 20 == 0:
            print(f"  Step {step:3d} | Loss: {losses[step]:.4f} | "
                  f"Chosen↑: {log_ratio_chosen[step]:+.3f} | "
                  f"Rejected↓: {log_ratio_rejected[step]:+.3f}")

        tracker.log_metric("demo_loss", float(losses[step]), step=step)
        tracker.log_metric("demo_chosen_log_ratio", float(log_ratio_chosen[step]), step=step)
        tracker.log_metric("demo_rejected_log_ratio", float(log_ratio_rejected[step]), step=step)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图 1: DPO 损失曲线
    axes[0].plot(losses, "b-", alpha=0.7)
    axes[0].set_title("DPO 损失曲线（模拟）")
    axes[0].set_xlabel("训练步数")
    axes[0].set_ylabel("损失")
    axes[0].grid(True, alpha=0.3)

    # 图 2: Chosen vs Rejected 概率比变化
    axes[1].plot(log_ratio_chosen, "g-", label="Chosen (↑)", alpha=0.7)
    axes[1].plot(log_ratio_rejected, "r-", label="Rejected (↓)", alpha=0.7)
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[1].set_title("Chosen vs Rejected 对数概率比")
    axes[1].set_xlabel("训练步数")
    axes[1].set_ylabel("log π_θ / π_ref")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 图 3: β 对 DPO 损失的影响
    betas = [0.01, 0.05, 0.1, 0.2, 0.5]
    x = np.linspace(-3, 3, 200)
    for b in betas:
        y = -np.log(1 / (1 + np.exp(-b * x)) + 1e-10)
        axes[2].plot(x, y, label=f"β={b}")
    axes[2].set_title("β 对 DPO 损失的影响")
    axes[2].set_xlabel("隐式奖励差 (r_chosen - r_rejected)")
    axes[2].set_ylabel("DPO 损失")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = PROJECT_ROOT / "day07_dpo" / "dpo_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📈 可视化结果已保存: {save_path}")

    tracker.log_text("mode", "demo_simulation")


if __name__ == "__main__":
    main()
