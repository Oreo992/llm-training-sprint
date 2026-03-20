"""
Day 08 - 对齐技术全景：从 RLHF 到 SimPO
==========================================

本脚本不进行实际训练，而是用代码和可视化展示各种对齐技术的：
- 核心思想和数学公式
- 演进关系和对比
- 各自的优缺点和适用场景

对齐技术演进路线：
    RLHF (2022) → DPO (2023) → 各种变体 (2023-2024)
                                 ├── KTO (Kahneman-Tversky Optimization)
                                 ├── ORPO (Odds Ratio Preference Optimization)
                                 ├── SimPO (Simple Preference Optimization)
                                 ├── IPO (Identity Preference Optimization)
                                 └── GRPO (Group Relative Policy Optimization)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from utils.experiment_tracker import ExperimentTracker

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ======================================================================
# 对齐技术定义
# ======================================================================

ALIGNMENT_METHODS = {
    "RLHF": {
        "full_name": "Reinforcement Learning from Human Feedback",
        "year": 2022,
        "paper": "Ouyang et al., 2022 (InstructGPT)",
        "steps": ["1. SFT 微调", "2. 训练奖励模型 (RM)", "3. PPO 强化学习优化"],
        "loss": "L_PPO = E[min(r_t·A_t, clip(r_t,1±ε)·A_t)] - β·KL(π||π_ref)",
        "pros": ["理论基础扎实", "效果经过大规模验证 (GPT-4, Claude)"],
        "cons": ["流程复杂（3 阶段）", "PPO 训练不稳定", "需要大量计算资源", "需要奖励模型"],
        "data_need": "偏好对 (用于训练 RM) + prompt (用于 PPO)",
        "category": "RL-based",
    },
    "DPO": {
        "full_name": "Direct Preference Optimization",
        "year": 2023,
        "paper": "Rafailov et al., 2023",
        "steps": ["1. SFT 微调", "2. 直接用偏好数据优化（无需 RM 和 PPO）"],
        "loss": "L_DPO = -E[log σ(β·(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]",
        "pros": ["简单稳定", "无需奖励模型", "无需 PPO", "闭式解"],
        "cons": ["需要参考模型 (内存翻倍)", "对数据质量敏感", "可能过拟合偏好数据"],
        "data_need": "偏好对 (prompt, chosen, rejected)",
        "category": "Preference-based",
    },
    "KTO": {
        "full_name": "Kahneman-Tversky Optimization",
        "year": 2024,
        "paper": "Ethayarajh et al., 2024",
        "steps": ["1. SFT 微调", "2. 用好/坏标签数据优化（不需要配对）"],
        "loss": "L_KTO = E_w[1-σ(β·r_w)] + E_l[1-σ(-β·r_l)]  (前景理论加权)",
        "pros": ["不需要配对偏好数据", "只需好/坏标签", "数据利用率更高"],
        "cons": ["理论较新，验证不如 DPO 充分", "超参数敏感"],
        "data_need": "单条标注 (prompt, response, good/bad)",
        "category": "Preference-based",
    },
    "ORPO": {
        "full_name": "Odds Ratio Preference Optimization",
        "year": 2024,
        "paper": "Hong et al., 2024",
        "steps": ["1. 直接在偏好数据上训练（SFT + 对齐一步完成）"],
        "loss": "L_ORPO = L_SFT + λ·L_OR  (SFT损失 + 赔率比损失)",
        "pros": ["不需要参考模型", "SFT 和对齐一步完成", "节省内存和计算"],
        "cons": ["可能不如两阶段方法精细", "超参数 λ 需要调节"],
        "data_need": "偏好对 (prompt, chosen, rejected)",
        "category": "Preference-based",
    },
    "SimPO": {
        "full_name": "Simple Preference Optimization",
        "year": 2024,
        "paper": "Meng et al., 2024",
        "steps": ["1. SFT 微调", "2. 用长度归一化的隐式奖励优化"],
        "loss": "L_SimPO = -E[log σ(β/|y_w|·log π_θ(y_w|x) - β/|y_l|·log π_θ(y_l|x) - γ)]",
        "pros": ["不需要参考模型", "长度归一化消除长度偏差", "性能优于 DPO"],
        "cons": ["引入额外超参数 γ (margin)"],
        "data_need": "偏好对 (prompt, chosen, rejected)",
        "category": "Preference-based",
    },
    "IPO": {
        "full_name": "Identity Preference Optimization",
        "year": 2024,
        "paper": "Azar et al., 2024",
        "steps": ["1. SFT 微调", "2. 用正则化偏好优化"],
        "loss": "L_IPO = E[(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x) - 1/2β)²]",
        "pros": ["避免 DPO 的过拟合问题", "理论上更健壮"],
        "cons": ["需要参考模型", "实际效果提升有限"],
        "data_need": "偏好对 (prompt, chosen, rejected)",
        "category": "Preference-based",
    },
    "GRPO": {
        "full_name": "Group Relative Policy Optimization",
        "year": 2024,
        "paper": "DeepSeek-R1, Shao et al., 2024",
        "steps": ["1. SFT 微调（可选）", "2. 对每个 prompt 采样多个回复", "3. 组内排名作为奖励信号"],
        "loss": "L_GRPO = -E[Σ min(r_t·Â_t, clip(r_t)·Â_t)] + β·KL  (Â=组内归一化优势)",
        "pros": ["不需要 critic 模型", "适合数学推理等可验证任务", "DeepSeek-R1 验证"],
        "cons": ["需要可靠的奖励函数", "采样多个回复增加推理成本"],
        "data_need": "prompt + 奖励函数（非偏好对）",
        "category": "RL-based",
    },
}


def print_comparison_table(tracker):
    """打印对齐技术对比表。"""
    print("\n" + "=" * 80)
    print("📊 对齐技术对比一览表")
    print("=" * 80)

    header = f"{'方法':<8} {'年份':<6} {'需要RM':<8} {'需要Ref':<8} {'训练阶段':<10} {'数据需求':<15}"
    print(header)
    print("-" * 80)

    for name, info in ALIGNMENT_METHODS.items():
        needs_rm = "是" if name == "RLHF" else "否"
        needs_ref = "是" if name in ["RLHF", "DPO", "IPO"] else "否"
        stages = str(len(info["steps"]))
        data = "偏好对" if "偏好对" in info["data_need"] else ("好/坏标签" if name == "KTO" else "prompt+reward")
        row = f"{name:<8} {info['year']:<6} {needs_rm:<8} {needs_ref:<8} {stages + '阶段':<10} {data:<15}"
        print(row)
        tracker.log_text(f"method_{name}", json.dumps(info, ensure_ascii=False))

    print()


def draw_evolution_graph(tracker):
    """
    绘制对齐技术演进关系图。
    使用 matplotlib 绘制节点和箭头展示技术之间的继承/改进关系。
    """
    print("\n🎨 绘制对齐技术演进图...")

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 9)
    ax.axis("off")
    ax.set_title("LLM 对齐技术演进全景图", fontsize=18, fontweight="bold", pad=20)

    # 节点定义：(x, y, 名称, 颜色, 描述)
    nodes = {
        "RLHF":  (1,  7, "RLHF\n(2022)", "#FF6B6B", "SFT→RM→PPO\n三阶段流程"),
        "DPO":   (4,  7, "DPO\n(2023)", "#4ECDC4", "直接偏好优化\n无需RM和PPO"),
        "KTO":   (2,  4, "KTO\n(2024)", "#45B7D1", "单条标注\n前景理论"),
        "ORPO":  (4.5, 4, "ORPO\n(2024)", "#96CEB4", "SFT+对齐\n一步完成"),
        "SimPO": (7,  4, "SimPO\n(2024)", "#FFEAA7", "长度归一化\n无需Ref模型"),
        "IPO":   (9,  7, "IPO\n(2024)", "#DDA0DD", "正则化DPO\n防过拟合"),
        "GRPO":  (7,  7, "GRPO\n(2024)", "#FF8C42", "组内排名奖励\nDeepSeek-R1"),
    }

    # 绘制节点
    for key, (x, y, label, color, desc) in nodes.items():
        # 主节点圆
        circle = plt.Circle((x, y), 0.8, color=color, alpha=0.85, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y + 0.1, label, ha="center", va="center", fontsize=9,
                fontweight="bold", zorder=4)
        # 描述文字
        ax.text(x, y - 1.3, desc, ha="center", va="top", fontsize=7,
                style="italic", color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # 演进箭头：(from, to, 标签, 样式)
    edges = [
        ("RLHF", "DPO",   "去除RM+PPO",    "-"),
        ("DPO",  "KTO",   "去除配对需求",   "--"),
        ("DPO",  "ORPO",  "去除Ref模型",    "--"),
        ("DPO",  "SimPO", "长度归一化",     "--"),
        ("DPO",  "IPO",   "正则化改进",     "--"),
        ("RLHF", "GRPO",  "去除Critic",    "-"),
    ]

    for src, dst, label, style in edges:
        sx, sy = nodes[src][0], nodes[src][1]
        dx, dy = nodes[dst][0], nodes[dst][1]
        ax.annotate(
            "", xy=(dx, dy + 0.8 if dy < sy else dy - 0.8),
            xytext=(sx, sy - 0.8 if dy < sy else sy + 0.8),
            arrowprops=dict(arrowstyle="->", color="#666666",
                           lw=1.5, linestyle=style, connectionstyle="arc3,rad=0.1"),
            zorder=2,
        )
        # 箭头标签
        mx, my = (sx + dx) / 2, (sy + dy) / 2
        ax.text(mx, my, label, ha="center", va="center", fontsize=7,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="#cccccc"))

    # 分类标签
    # RL-based 区域
    rect1 = mpatches.FancyBboxPatch((-0.3, 5.8), 9, 2.8, boxstyle="round,pad=0.3",
                                      facecolor="#FFE0E0", alpha=0.2, edgecolor="#FF6B6B",
                                      linestyle="--", zorder=1)
    ax.add_patch(rect1)
    ax.text(-0.1, 8.4, "RL-based 方法", fontsize=10, color="#FF6B6B", fontweight="bold")

    # Preference-based 区域
    rect2 = mpatches.FancyBboxPatch((0.5, 2.5), 8, 2.3, boxstyle="round,pad=0.3",
                                      facecolor="#E0FFE0", alpha=0.2, edgecolor="#4ECDC4",
                                      linestyle="--", zorder=1)
    ax.add_patch(rect2)
    ax.text(0.7, 4.6, "Preference-based 方法（DPO 变体）", fontsize=10,
            color="#4ECDC4", fontweight="bold")

    # 时间轴
    ax.annotate("", xy=(10, 0.5), xytext=(0, 0.5),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2))
    ax.text(5, 0.1, "时间演进 →", ha="center", fontsize=10, color="gray")
    for year, x_pos in [(2022, 1), (2023, 4), (2024, 7.5)]:
        ax.text(x_pos, 0.6, str(year), ha="center", fontsize=9, color="gray", fontweight="bold")

    plt.tight_layout()
    save_path = PROJECT_ROOT / "day08_alignment_landscape" / "alignment_landscape.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 演进图已保存: {save_path}")
    tracker.log_text("landscape_image", str(save_path))


def draw_loss_comparison(tracker):
    """用数值模拟对比各种对齐方法的损失函数形状。"""
    print("\n📐 绘制损失函数对比...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.linspace(-4, 4, 500)  # 隐式奖励差

    # 图 1: DPO vs IPO 损失
    ax = axes[0, 0]
    beta = 0.1
    # DPO: -log σ(β·x)
    dpo_loss = -np.log(1 / (1 + np.exp(-beta * x)) + 1e-10)
    # IPO: (β·x - 1/(2β))²
    ipo_loss = (x - 1 / (2 * beta)) ** 2
    ipo_loss = ipo_loss / ipo_loss.max() * dpo_loss.max()  # 归一化显示

    ax.plot(x, dpo_loss, "b-", label="DPO", linewidth=2)
    ax.plot(x, ipo_loss, "r--", label="IPO (归一化)", linewidth=2)
    ax.set_title("DPO vs IPO 损失函数")
    ax.set_xlabel("隐式奖励差 (chosen - rejected)")
    ax.set_ylabel("损失")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    # 图 2: SimPO 长度归一化效果
    ax = axes[0, 1]
    lengths = [10, 50, 100, 200]
    for l in lengths:
        simpo_loss = -np.log(1 / (1 + np.exp(-(beta / l) * x * l)) + 1e-10)
        ax.plot(x, simpo_loss, label=f"长度={l}", linewidth=1.5)
    ax.set_title("SimPO: 长度归一化消除偏差")
    ax.set_xlabel("原始对数概率差")
    ax.set_ylabel("损失")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图 3: KTO 好/坏样本分别的损失
    ax = axes[1, 0]
    r = np.linspace(-3, 3, 500)
    kto_good = 1 - 1 / (1 + np.exp(-beta * r))   # 好样本：希望 r 高
    kto_bad = 1 - 1 / (1 + np.exp(beta * r))      # 坏样本：希望 r 低
    ax.plot(r, kto_good, "g-", label="好样本损失 (希望r↑)", linewidth=2)
    ax.plot(r, kto_bad, "r-", label="坏样本损失 (希望r↓)", linewidth=2)
    ax.set_title("KTO: 好/坏样本的独立损失")
    ax.set_xlabel("隐式奖励 r(x,y)")
    ax.set_ylabel("损失")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    # 图 4: 方法复杂度对比柱状图
    ax = axes[1, 1]
    methods = ["RLHF", "DPO", "KTO", "ORPO", "SimPO", "GRPO"]
    complexity = [5, 3, 2, 2, 2, 4]       # 实现复杂度 (1-5)
    memory = [5, 4, 3, 2, 2, 3]           # 内存需求 (1-5)
    data_req = [4, 3, 2, 3, 3, 2]         # 数据需求 (1-5)

    x_pos = np.arange(len(methods))
    width = 0.25
    ax.bar(x_pos - width, complexity, width, label="实现复杂度", color="#FF6B6B", alpha=0.8)
    ax.bar(x_pos, memory, width, label="内存需求", color="#4ECDC4", alpha=0.8)
    ax.bar(x_pos + width, data_req, width, label="数据需求", color="#45B7D1", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_title("对齐方法资源需求对比")
    ax.set_ylabel("相对程度 (1-5)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = PROJECT_ROOT / "day08_alignment_landscape" / "loss_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 损失函数对比图已保存: {save_path}")

    # 记录对比数据到 tracker
    for i, m in enumerate(methods):
        tracker.log_metric(f"{m}_complexity", complexity[i])
        tracker.log_metric(f"{m}_memory", memory[i])
        tracker.log_metric(f"{m}_data_requirement", data_req[i])


def print_detailed_analysis(tracker):
    """打印每种方法的详细分析。"""
    print("\n" + "=" * 80)
    print("📖 各方法详细分析")
    print("=" * 80)

    for name, info in ALIGNMENT_METHODS.items():
        print(f"\n{'─' * 60}")
        print(f"  {name} ({info['full_name']})")
        print(f"  论文: {info['paper']}")
        print(f"{'─' * 60}")
        print(f"  训练流程:")
        for step in info["steps"]:
            print(f"    {step}")
        print(f"\n  损失函数:")
        print(f"    {info['loss']}")
        print(f"\n  优点:")
        for p in info["pros"]:
            print(f"    ✅ {p}")
        print(f"  缺点:")
        for c in info["cons"]:
            print(f"    ❌ {c}")
        print(f"\n  数据需求: {info['data_need']}")

    tracker.log_text("analysis_complete", "true")


import json

def main():
    print("=" * 60)
    print("Day 08 - 对齐技术全景：从 RLHF 到 SimPO")
    print("=" * 60)

    tracker = ExperimentTracker(
        experiment_name="day08_alignment_landscape",
        tags={"day": "08", "task": "alignment_overview"},
    )

    # 1. 打印对比表
    print_comparison_table(tracker)

    # 2. 绘制演进图
    draw_evolution_graph(tracker)

    # 3. 绘制损失函数对比
    draw_loss_comparison(tracker)

    # 4. 详细分析
    print_detailed_analysis(tracker)

    # 5. 记录方法数量
    tracker.log_metric("num_methods_covered", len(ALIGNMENT_METHODS))

    print("\n" + "=" * 60)
    print("📋 实验摘要")
    print("=" * 60)
    summary = tracker.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n✅ 完成！图像保存在 day08_alignment_landscape/ 目录下。")


if __name__ == "__main__":
    main()
