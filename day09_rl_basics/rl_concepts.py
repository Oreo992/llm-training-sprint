"""
Day 09 - RL 基础概念可视化：从经典 RL 到 LLM
==============================================

核心映射关系：
    经典 RL                    LLM 对齐
    ────────────────────────────────────────
    State (状态)      ←→     已生成的 token 序列 (prompt + 已生成部分)
    Action (动作)     ←→     选择下一个 token（词表中选一个）
    Policy (策略)     ←→     语言模型本身 π_θ(a_t | s_t)
    Reward (奖励)     ←→     人类偏好评分 / 奖励模型输出
    Episode (回合)    ←→     生成一个完整回复
    Trajectory (轨迹) ←→     prompt → token1 → token2 → ... → EOS
    Value (价值)      ←→     从当前位置继续生成能获得的期望总奖励

    REINFORCE 算法核心思想：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ∇J(θ) = E[Σ_t ∇log π_θ(a_t|s_t) · G_t]

    直觉：
    - 如果一个动作序列获得了高回报 G_t → 提高这些动作的概率
    - 如果一个动作序列获得了低回报 G_t → 降低这些动作的概率
    - log π_θ(a_t|s_t) 就是 LLM 对 next token 的对数概率
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.experiment_tracker import ExperimentTracker

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ======================================================================
# Toy Environment: 多臂赌博机 (Multi-Armed Bandit)
# 类比：LLM 在给定 prompt 下选择不同回复策略
# ======================================================================

class ToyBanditEnv:
    """
    多臂赌博机环境。
    类比 LLM 场景：每个 "臂" 对应一种回复风格，
    奖励代表人类对该风格回复的满意度。

    臂 0: "简短直接" 风格 → 中等奖励
    臂 1: "详细解释" 风格 → 高奖励
    臂 2: "废话连篇" 风格 → 低奖励
    臂 3: "创意发散" 风格 → 奖励波动大
    """

    def __init__(self, n_arms=4):
        self.n_arms = n_arms
        # 每个臂的真实奖励均值（模拟人类偏好）
        self.true_means = [0.5, 0.8, 0.2, 0.6]
        # 奖励标准差
        self.true_stds = [0.1, 0.15, 0.1, 0.4]
        self.labels = ["简短直接", "详细解释", "废话连篇", "创意发散"]

    def step(self, action):
        """执行动作，返回奖励。"""
        reward = np.random.normal(self.true_means[action], self.true_stds[action])
        return np.clip(reward, 0, 1)  # 奖励限制在 [0,1]


# ======================================================================
# REINFORCE 算法实现
# ======================================================================

class REINFORCEAgent:
    """
    REINFORCE (策略梯度) 算法。

    这是最基础的策略梯度方法，也是 RLHF 中 PPO 的前身。
    核心思想：
        1. 用当前策略采样动作
        2. 观察奖励
        3. 如果奖励高 → 增加该动作的概率
        4. 如果奖励低 → 降低该动作的概率

    数学：
        π_θ(a) = softmax(θ)  （策略 = θ 的 softmax）
        ∇J = (R - baseline) · ∇log π_θ(a)  （策略梯度）
        θ ← θ + α · ∇J  （参数更新）

    在 LLM 中：
        θ = 模型参数
        a = 生成的 token
        R = 奖励模型的评分
    """

    def __init__(self, n_actions, lr=0.1, baseline_decay=0.9):
        # 策略参数（类比 LLM 的模型权重）
        self.theta = np.zeros(n_actions)
        self.lr = lr
        # 基线：用移动平均减少梯度方差
        # 类比 LLM RLHF 中的 value function baseline
        self.baseline = 0.0
        self.baseline_decay = baseline_decay

    def get_policy(self):
        """计算当前策略（softmax）。类比 LLM 的 next token 概率分布。"""
        exp_theta = np.exp(self.theta - np.max(self.theta))
        return exp_theta / exp_theta.sum()

    def select_action(self):
        """根据当前策略采样动作。类比 LLM 采样下一个 token。"""
        probs = self.get_policy()
        return np.random.choice(len(probs), p=probs)

    def update(self, action, reward):
        """
        REINFORCE 更新。

        关键步骤：
        1. 计算优势 = reward - baseline（减少方差）
        2. 计算策略梯度 ∇log π_θ(a)
        3. 沿梯度方向更新参数

        在 LLM RLHF 中：
        - advantage = R(y|x) - V(x)  (奖励 - 价值函数基线)
        - 对整个 token 序列计算梯度
        """
        advantage = reward - self.baseline

        # 更新基线（移动平均）
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward

        # 策略梯度更新
        probs = self.get_policy()
        # ∇log π_θ(a) = e_a - π_θ  （one-hot 减去概率分布）
        grad_log_pi = np.zeros_like(self.theta)
        grad_log_pi[action] = 1.0
        grad_log_pi -= probs

        # θ ← θ + α · advantage · ∇log π_θ(a)
        self.theta += self.lr * advantage * grad_log_pi


def run_reinforce_experiment(tracker, n_episodes=500):
    """运行 REINFORCE 实验并记录指标。"""
    print("\n" + "=" * 60)
    print("🎰 REINFORCE 实验：多臂赌博机")
    print("=" * 60)

    env = ToyBanditEnv()
    agent = REINFORCEAgent(n_actions=env.n_arms, lr=0.15)

    rewards_history = []
    policy_history = []
    action_history = []

    for episode in range(n_episodes):
        action = agent.select_action()
        reward = env.step(action)
        agent.update(action, reward)

        rewards_history.append(reward)
        policy_history.append(agent.get_policy().copy())
        action_history.append(action)

        tracker.log_metric("reward", float(reward), step=episode)

        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
            policy = agent.get_policy()
            print(f"  Episode {episode:4d} | 平均奖励: {avg_reward:.3f} | "
                  f"策略: [{', '.join(f'{p:.2f}' for p in policy)}]")

    # 最终结果
    final_policy = agent.get_policy()
    best_action = np.argmax(final_policy)
    print(f"\n  ✅ 最终策略: [{', '.join(f'{p:.3f}' for p in final_policy)}]")
    print(f"  最优动作: {env.labels[best_action]} (概率 {final_policy[best_action]:.3f})")
    print(f"  真实最优: {env.labels[np.argmax(env.true_means)]} (均值 {max(env.true_means):.3f})")

    tracker.log_metric("final_best_action_prob", float(final_policy[best_action]))
    tracker.log_text("final_policy", str(final_policy.tolist()))
    tracker.log_text("best_action", env.labels[best_action])

    return rewards_history, policy_history, action_history, env


def visualize_rl_concepts(rewards, policies, actions, env, tracker):
    """可视化 RL 核心概念。"""
    print("\n📊 生成可视化...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图 1: 奖励曲线（滑动平均）
    ax = axes[0, 0]
    window = 20
    rewards_arr = np.array(rewards)
    smoothed = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
    ax.plot(smoothed, "b-", alpha=0.8, linewidth=1.5)
    ax.axhline(y=max(env.true_means), color="r", linestyle="--", alpha=0.5,
               label=f"最优奖励 ({max(env.true_means):.2f})")
    ax.set_title("REINFORCE 奖励曲线")
    ax.set_xlabel("Episode")
    ax.set_ylabel("奖励（滑动平均）")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图 2: 策略演变
    ax = axes[0, 1]
    policies_arr = np.array(policies)
    for i in range(env.n_arms):
        ax.plot(policies_arr[:, i], label=env.labels[i], linewidth=1.5)
    ax.set_title("策略演变：各动作概率随训练变化")
    ax.set_xlabel("Episode")
    ax.set_ylabel("动作概率 π(a)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图 3: RL → LLM 概念映射图
    ax = axes[1, 0]
    ax.axis("off")
    ax.set_title("RL ↔ LLM 对齐 概念映射", fontsize=13, fontweight="bold")

    mapping = [
        ("经典 RL 概念", "LLM 对齐中的对应"),
        ("───────────", "──────────────"),
        ("State (状态)", "已生成的 token 序列"),
        ("Action (动作)", "选择下一个 token"),
        ("Policy π(a|s)", "语言模型 LM(token|context)"),
        ("Reward R", "人类偏好评分 / RM 输出"),
        ("Episode", "生成一个完整回复"),
        ("Trajectory", "prompt→t1→t2→...→EOS"),
        ("Value V(s)", "从当前位置的期望回报"),
        ("Advantage A", "R(y|x) - V(x)"),
        ("REINFORCE", "基础策略梯度"),
        ("PPO", "RLHF 中的策略优化"),
        ("GRPO", "无 Critic 的组内排名"),
    ]
    text = "\n".join(f"  {left:<22s} ←→  {right}" for left, right in mapping)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    # 图 4: 动作选择频率分布
    ax = axes[1, 1]
    # 前 100 次 vs 后 100 次
    early_actions = actions[:100]
    late_actions = actions[-100:]
    x_pos = np.arange(env.n_arms)
    width = 0.35

    early_counts = [early_actions.count(i) / 100 for i in range(env.n_arms)]
    late_counts = [late_actions.count(i) / 100 for i in range(env.n_arms)]

    ax.bar(x_pos - width / 2, early_counts, width, label="前 100 次", color="#FF6B6B", alpha=0.8)
    ax.bar(x_pos + width / 2, late_counts, width, label="后 100 次", color="#4ECDC4", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(env.labels)
    ax.set_title("动作选择频率：探索 → 利用")
    ax.set_ylabel("选择频率")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = PROJECT_ROOT / "day09_rl_basics" / "rl_concepts.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 可视化已保存: {save_path}")
    tracker.log_text("visualization", str(save_path))


def visualize_policy_gradient_intuition(tracker):
    """
    可视化策略梯度的直觉：
    高奖励 → 增加概率，低奖励 → 降低概率
    """
    print("\n📐 策略梯度直觉可视化...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 模拟一个简单的 3-token 词表
    tokens = ["好", "一般", "差"]

    # 初始概率分布（均匀）
    initial_probs = [1 / 3, 1 / 3, 1 / 3]
    # 奖励信号
    rewards = [0.9, 0.5, 0.1]
    # 更新后的概率（模拟多轮 REINFORCE）
    updated_probs = [0.6, 0.3, 0.1]

    # 图 1: 初始策略
    axes[0].bar(tokens, initial_probs, color=["#4ECDC4", "#FFD93D", "#FF6B6B"], alpha=0.8)
    axes[0].set_title("初始策略 π₀(token)")
    axes[0].set_ylabel("概率")
    axes[0].set_ylim(0, 0.8)
    for i, v in enumerate(initial_probs):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)

    # 图 2: 奖励信号
    colors = ["#4ECDC4" if r > 0.5 else "#FFD93D" if r > 0.3 else "#FF6B6B" for r in rewards]
    axes[1].bar(tokens, rewards, color=colors, alpha=0.8)
    axes[1].set_title("奖励信号 R(token)")
    axes[1].set_ylabel("奖励")
    axes[1].set_ylim(0, 1.1)
    for i, v in enumerate(rewards):
        axes[1].text(i, v + 0.03, f"{v:.1f}", ha="center", fontsize=11)
    axes[1].axhline(y=np.mean(rewards), color="gray", linestyle="--", alpha=0.5,
                     label=f"基线 b={np.mean(rewards):.2f}")
    axes[1].legend()

    # 图 3: 更新后策略
    axes[2].bar(tokens, updated_probs, color=["#4ECDC4", "#FFD93D", "#FF6B6B"], alpha=0.8)
    axes[2].set_title("更新后策略 π_θ(token)")
    axes[2].set_ylabel("概率")
    axes[2].set_ylim(0, 0.8)
    for i, v in enumerate(updated_probs):
        diff = v - initial_probs[i]
        sign = "↑" if diff > 0 else "↓"
        axes[2].text(i, v + 0.02, f"{v:.2f} ({sign}{abs(diff):.2f})", ha="center", fontsize=10)

    plt.suptitle("策略梯度直觉：高奖励→概率↑  低奖励→概率↓", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = PROJECT_ROOT / "day09_rl_basics" / "policy_gradient_intuition.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 策略梯度直觉图已保存: {save_path}")


def main():
    print("=" * 60)
    print("Day 09 - RL 基础概念可视化")
    print("=" * 60)

    np.random.seed(42)

    tracker = ExperimentTracker(
        experiment_name="day09_rl_concepts",
        tags={"day": "09", "task": "rl_basics", "method": "REINFORCE"},
    )

    # 1. 运行 REINFORCE 实验
    rewards, policies, actions, env = run_reinforce_experiment(tracker, n_episodes=500)

    # 2. 可视化实验结果
    visualize_rl_concepts(rewards, policies, actions, env, tracker)

    # 3. 策略梯度直觉可视化
    visualize_policy_gradient_intuition(tracker)

    # 4. 打印实验摘要
    print("\n" + "=" * 60)
    print("📋 实验摘要")
    print("=" * 60)
    summary = tracker.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n✅ 完成！可视化结果保存在 day09_rl_basics/ 目录下。")


if __name__ == "__main__":
    main()
