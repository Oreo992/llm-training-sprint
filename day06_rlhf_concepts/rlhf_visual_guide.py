"""
RLHF 概念可视化指南
==================

用 matplotlib 可视化 RLHF（Reinforcement Learning from Human Feedback）的核心概念：

1. RLHF 经典三阶段流程图：SFT → Reward Model → PPO
2. Reward Model 工作原理示意图
3. PPO 的 KL 约束直觉可视化
4. DPO vs RLHF 对比

核心概念解释：
- RLHF 的核心思想：用人类偏好来指导模型训练
- SFT：先用监督学习让模型学会基本格式
- Reward Model：学习人类偏好，给回复打分
- PPO：用奖励信号优化策略，同时用 KL 散度约束防止偏离太远
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def draw_rlhf_pipeline(ax):
    """
    绘制 RLHF 经典三阶段流程图。

    RLHF 流程：
    Stage 1: SFT（有监督微调）
      - 输入：预训练模型 + 高质量标注数据
      - 输出：SFT 模型（学会了基本的对话格式和指令遵循）

    Stage 2: Reward Model Training（奖励模型训练）
      - 输入：SFT 模型生成的多个回复 + 人类排序标注
      - 输出：奖励模型（能够评分回复质量）
      - 关键：Bradley-Terry 模型，r(y_w) > r(y_l)

    Stage 3: PPO（策略优化）
      - 输入：SFT 模型（初始策略）+ 奖励模型
      - 输出：对齐后的模型
      - 关键：最大化奖励 - β * KL(π || π_ref)
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("RLHF 三阶段流程", fontsize=14, fontweight="bold", pad=15)

    # 颜色方案
    colors = {
        "stage1": "#4ECDC4",  # 青色
        "stage2": "#FF6B6B",  # 红色
        "stage3": "#45B7D1",  # 蓝色
        "data": "#FFE66D",    # 黄色
        "model": "#95E1D3",   # 浅绿
    }

    # Stage 1: SFT
    box1 = FancyBboxPatch((0.5, 3.5), 2.5, 1.8, boxstyle="round,pad=0.15",
                           facecolor=colors["stage1"], edgecolor="black", linewidth=1.5, alpha=0.85)
    ax.add_patch(box1)
    ax.text(1.75, 4.9, "Stage 1: SFT", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(1.75, 4.4, "有监督微调", ha="center", va="center", fontsize=9)
    ax.text(1.75, 3.9, "预训练模型 + 标注数据", ha="center", va="center", fontsize=8, style="italic")

    # Stage 2: Reward Model
    box2 = FancyBboxPatch((3.7, 3.5), 2.5, 1.8, boxstyle="round,pad=0.15",
                           facecolor=colors["stage2"], edgecolor="black", linewidth=1.5, alpha=0.85)
    ax.add_patch(box2)
    ax.text(4.95, 4.9, "Stage 2: RM", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(4.95, 4.4, "奖励模型训练", ha="center", va="center", fontsize=9)
    ax.text(4.95, 3.9, "人类偏好排序数据", ha="center", va="center", fontsize=8, style="italic")

    # Stage 3: PPO
    box3 = FancyBboxPatch((6.9, 3.5), 2.5, 1.8, boxstyle="round,pad=0.15",
                           facecolor=colors["stage3"], edgecolor="black", linewidth=1.5, alpha=0.85)
    ax.add_patch(box3)
    ax.text(8.15, 4.9, "Stage 3: PPO", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(8.15, 4.4, "策略优化", ha="center", va="center", fontsize=9)
    ax.text(8.15, 3.9, "max R(y) - β·KL", ha="center", va="center", fontsize=8, style="italic")

    # 箭头
    ax.annotate("", xy=(3.7, 4.4), xytext=(3.0, 4.4),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.annotate("", xy=(6.9, 4.4), xytext=(6.2, 4.4),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))

    # 输入输出标注
    # 底部数据流
    data_items = [
        (1.75, 2.5, "标注对话\n数据集", colors["data"]),
        (4.95, 2.5, "偏好对\n(y_w, y_l)", colors["data"]),
        (8.15, 2.5, "RM 评分\n+ KL 约束", colors["data"]),
    ]
    for x, y, text, color in data_items:
        box = FancyBboxPatch((x - 0.9, y - 0.5), 1.8, 1.0, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="gray", linewidth=1, alpha=0.7)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5)
        ax.annotate("", xy=(x, 3.5), xytext=(x, y + 0.5),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1, ls="--"))

    # 输出模型标注
    ax.text(1.75, 1.2, "→ SFT 模型", ha="center", fontsize=8, color=colors["stage1"], fontweight="bold")
    ax.text(4.95, 1.2, "→ 奖励模型", ha="center", fontsize=8, color=colors["stage2"], fontweight="bold")
    ax.text(8.15, 1.2, "→ 对齐模型", ha="center", fontsize=8, color=colors["stage3"], fontweight="bold")


def draw_reward_model(ax):
    """
    绘制 Reward Model 工作原理。

    奖励模型的核心：
    1. 输入一个 prompt + response，输出一个标量奖励值
    2. 训练目标：让好回复的奖励值 > 坏回复的奖励值
    3. 损失函数（Bradley-Terry 模型）：
       L = -log σ(r(y_w) - r(y_l))
       其中 y_w 是人类偏好的回复，y_l 是不被偏好的回复
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Reward Model 工作原理", fontsize=14, fontweight="bold", pad=15)

    # Prompt
    prompt_box = FancyBboxPatch((0.3, 4.0), 2.0, 1.2, boxstyle="round,pad=0.1",
                                 facecolor="#FFE66D", edgecolor="black", linewidth=1.5)
    ax.add_patch(prompt_box)
    ax.text(1.3, 4.6, "Prompt", ha="center", va="center", fontsize=10, fontweight="bold")

    # 两个 Response
    # Good response
    good_box = FancyBboxPatch((3.5, 4.5), 2.5, 1.0, boxstyle="round,pad=0.1",
                               facecolor="#95E1D3", edgecolor="green", linewidth=1.5)
    ax.add_patch(good_box)
    ax.text(4.75, 5.0, "Response A (好)", ha="center", va="center", fontsize=9, color="green")

    # Bad response
    bad_box = FancyBboxPatch((3.5, 3.0), 2.5, 1.0, boxstyle="round,pad=0.1",
                              facecolor="#FFB3B3", edgecolor="red", linewidth=1.5)
    ax.add_patch(bad_box)
    ax.text(4.75, 3.5, "Response B (差)", ha="center", va="center", fontsize=9, color="red")

    # 箭头从 prompt 到 responses
    ax.annotate("", xy=(3.5, 5.0), xytext=(2.3, 4.8),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate("", xy=(3.5, 3.5), xytext=(2.3, 4.2),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Reward Model
    rm_box = FancyBboxPatch((7.0, 3.5), 2.2, 2.0, boxstyle="round,pad=0.15",
                             facecolor="#45B7D1", edgecolor="black", linewidth=1.5, alpha=0.85)
    ax.add_patch(rm_box)
    ax.text(8.1, 4.8, "Reward", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    ax.text(8.1, 4.3, "Model", ha="center", va="center", fontsize=10, fontweight="bold", color="white")

    # 奖励分数
    ax.text(8.1, 3.8, "r(A) = 0.85", ha="center", va="center", fontsize=9, color="green", fontweight="bold")
    ax.text(8.1, 3.4, "r(B) = 0.23", ha="center", va="center", fontsize=9, color="red", fontweight="bold")

    # 箭头到 RM
    ax.annotate("", xy=(7.0, 5.0), xytext=(6.0, 5.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate("", xy=(7.0, 3.5), xytext=(6.0, 3.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # 损失函数公式
    ax.text(5.0, 1.8, "训练目标（Bradley-Terry 模型）:", ha="center", fontsize=10, fontweight="bold")
    ax.text(5.0, 1.1, r"$\mathcal{L} = -\log\sigma(r(y_w) - r(y_l))$",
            ha="center", fontsize=12, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0", edgecolor="gray"))
    ax.text(5.0, 0.4, "目标：让好回复的奖励值 > 坏回复的奖励值", ha="center", fontsize=9, color="gray")


def draw_ppo_kl_constraint(ax):
    """
    绘制 PPO 的 KL 约束直觉。

    KL 散度约束的重要性：
    - 没有 KL 约束：模型会过度优化奖励模型的漏洞（reward hacking）
    - 合适的 KL 约束：在提升回复质量的同时保持模型的基本能力
    - 过强的 KL 约束：模型几乎不变，无法学到新的偏好

    PPO 目标函数：
    max E[R(y|x) - β · KL(π_θ || π_ref)]
    """
    ax.set_title("PPO 的 KL 约束直觉", fontsize=14, fontweight="bold", pad=15)

    # 模拟不同 β 值下的训练曲线
    steps = np.linspace(0, 100, 200)

    # 奖励曲线（不同 KL 约束强度）
    # β = 0（无约束）：奖励先升后可能出问题（reward hacking）
    reward_no_kl = 0.8 * (1 - np.exp(-steps / 20)) + 0.15 * np.sin(steps / 15)
    reward_no_kl[steps > 60] += 0.3 * (steps[steps > 60] - 60) / 40  # reward hacking

    # β = 0.01（弱约束）：稳定提升
    reward_weak_kl = 0.7 * (1 - np.exp(-steps / 25))

    # β = 0.1（适中约束）：最佳平衡
    reward_good_kl = 0.6 * (1 - np.exp(-steps / 30))

    # β = 1.0（强约束）：几乎不变
    reward_strong_kl = 0.15 * (1 - np.exp(-steps / 50))

    ax.plot(steps, reward_no_kl, color="#FF6B6B", linewidth=2, label="β=0 (无约束)", linestyle="--")
    ax.plot(steps, reward_weak_kl, color="#FFB347", linewidth=2, label="β=0.01 (弱约束)")
    ax.plot(steps, reward_good_kl, color="#4ECDC4", linewidth=2.5, label="β=0.1 (适中) ★")
    ax.plot(steps, reward_strong_kl, color="#95A5A6", linewidth=2, label="β=1.0 (强约束)")

    # 标注 reward hacking 区域
    ax.axvspan(60, 100, alpha=0.1, color="red")
    ax.text(80, 1.0, "Reward\nHacking", ha="center", va="center",
            fontsize=9, color="red", fontweight="bold", alpha=0.7)

    # 标注最优区域
    ax.axhline(y=0.6, color="gray", linestyle=":", alpha=0.3)
    ax.text(105, 0.6, "理想\n奖励", fontsize=8, color="gray", va="center")

    ax.set_xlabel("训练步数", fontsize=11)
    ax.set_ylabel("奖励值", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 110)
    ax.set_ylim(-0.1, 1.3)

    # 底部公式
    ax.text(55, -0.05, r"PPO 目标: $\max\ \mathbb{E}[R(y|x) - \beta \cdot KL(\pi_\theta \| \pi_{ref})]$",
            ha="center", fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F8F8", edgecolor="gray"))


def draw_kl_divergence_intuition(ax):
    """
    绘制 KL 散度的直觉理解。

    KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

    直觉：
    - KL = 0：两个分布完全相同
    - KL 越大：两个分布差异越大
    - KL 不对称：KL(P||Q) ≠ KL(Q||P)

    在 RLHF 中：
    - P = π_θ（当前策略）
    - Q = π_ref（参考策略/SFT 模型）
    - 限制 KL(π_θ || π_ref) 防止策略偏离太远
    """
    ax.set_title("KL 散度约束示意", fontsize=14, fontweight="bold", pad=15)

    x = np.linspace(-4, 6, 300)

    # 参考分布 (SFT 模型)
    ref_dist = np.exp(-0.5 * (x - 1) ** 2) / np.sqrt(2 * np.pi)

    # 不同程度偏移的策略分布
    policy_small = np.exp(-0.5 * (x - 1.5) ** 2 / 1.1 ** 2) / (1.1 * np.sqrt(2 * np.pi))
    policy_medium = np.exp(-0.5 * (x - 2.5) ** 2 / 1.3 ** 2) / (1.3 * np.sqrt(2 * np.pi))
    policy_large = np.exp(-0.5 * (x - 4) ** 2 / 1.5 ** 2) / (1.5 * np.sqrt(2 * np.pi))

    ax.fill_between(x, ref_dist, alpha=0.3, color="#4ECDC4")
    ax.plot(x, ref_dist, color="#4ECDC4", linewidth=2.5, label="π_ref (SFT模型)")

    ax.plot(x, policy_small, color="#45B7D1", linewidth=2, linestyle="--", label="π_θ 小偏移 (KL≈0.1)")
    ax.plot(x, policy_medium, color="#FFB347", linewidth=2, linestyle="-.", label="π_θ 中偏移 (KL≈0.5)")
    ax.plot(x, policy_large, color="#FF6B6B", linewidth=2, linestyle=":", label="π_θ 大偏移 (KL≈2.0)")

    # 标注 KL 约束边界
    ax.axvline(x=1, color="#4ECDC4", linestyle=":", alpha=0.5)
    ax.text(1, 0.42, "参考\n策略", ha="center", fontsize=8, color="#4ECDC4")

    # 允许的偏移范围
    ax.annotate("", xy=(2.5, 0.35), xytext=(1, 0.35),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
    ax.text(1.75, 0.37, "允许范围", ha="center", fontsize=8, color="green")

    ax.set_xlabel("输出空间", fontsize=11)
    ax.set_ylabel("概率密度", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.45)


def create_rlhf_visual_guide():
    """生成完整的 RLHF 概念可视化图"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("RLHF (Reinforcement Learning from Human Feedback) 概念可视化指南",
                 fontsize=18, fontweight="bold", y=0.98)

    draw_rlhf_pipeline(axes[0, 0])
    draw_reward_model(axes[0, 1])
    draw_ppo_kl_constraint(axes[1, 0])
    draw_kl_divergence_intuition(axes[1, 1])

    # 底部注释
    fig.text(0.5, 0.01,
             "RLHF 通过人类反馈训练奖励模型，再用 PPO 优化策略。KL 约束是防止 reward hacking 的关键。",
             ha="center", fontsize=11, style="italic", color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    output_path = os.path.join(os.path.dirname(__file__), "rlhf_concepts.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"RLHF 概念图已保存到: {output_path}")
    return output_path


if __name__ == "__main__":
    output = create_rlhf_visual_guide()
    print(f"\n完成！文件路径: {output}")
