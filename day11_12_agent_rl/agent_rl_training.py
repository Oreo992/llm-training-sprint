"""
Day 11-12 - AgentRL 训练：用 GRPO 训练文件操作 Agent
=====================================================

核心思想：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    将 Agent 的多轮交互视为一个完整的 "轨迹" (trajectory)：
        prompt → action_1 → obs_1 → action_2 → obs_2 → ... → done

    Trajectory-level Optimization:
    - 不是对每个 token 单独优化，而是对整个交互轨迹评分
    - 奖励在轨迹结束时给出（稀疏奖励）
    - 用 GRPO：对同一个任务采样多条轨迹，组内排名

    过程奖励 (Process Reward) vs 结果奖励 (Outcome Reward):
    ────────────────────────────────────────────────────────
    结果奖励: 只看最终结果是否正确
      + 简单直接
      - 稀疏信号，学习困难

    过程奖励: 对中间步骤也给奖励
      + 更密集的学习信号
      + 可以引导探索
      - 需要设计中间奖励（可能引入偏差）
      - 可能导致 reward hacking
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.experiment_tracker import ExperimentTracker
from day11_12_agent_rl.agent_env import AgentEnvironment, Task, create_tasks

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ======================================================================
# 模拟 Agent 策略（不同能力水平）
# ======================================================================

class RandomAgent:
    """随机 Agent：随机选择动作（基线）。"""

    def __init__(self, max_actions=5):
        self.max_actions = max_actions
        self.name = "RandomAgent"

    def act(self, observation: str, env: AgentEnvironment) -> str:
        """根据观测随机选择动作。"""
        actions = [
            f"list_dir /",
            f"search file",
            f"read_file /data/readme.md",
            f"done unknown",
        ]
        # 添加环境中实际存在的文件路径
        for path in list(env.fs.files.keys())[:3]:
            actions.append(f"read_file {path}")
            dir_path = "/".join(path.split("/")[:-1])
            if dir_path:
                actions.append(f"list_dir {dir_path}")

        return np.random.choice(actions)


class HeuristicAgent:
    """
    启发式 Agent：使用简单规则做决策。
    模拟训练中期的模型——比随机好，但不完美。
    """

    def __init__(self):
        self.name = "HeuristicAgent"
        self.step = 0
        self.found_files = []
        self.read_contents = {}

    def reset(self):
        self.step = 0
        self.found_files = []
        self.read_contents = {}

    def act(self, observation: str, env: AgentEnvironment) -> str:
        """基于启发式规则选择动作。"""
        self.step += 1
        task_desc = env.task.description.lower()

        # 第一步：列出相关目录
        if self.step == 1:
            # 从任务描述中提取目录路径
            dirs = re.findall(r"/\w+", task_desc)
            if dirs:
                return f"list_dir {dirs[0]}"
            return "list_dir /"

        # 第二步：搜索关键词
        if self.step == 2:
            keywords = ["error", "warning", "port", "数据库"]
            for kw in keywords:
                if kw in task_desc:
                    return f"search {kw}"
            return f"search {task_desc.split()[0]}"

        # 后续步骤：读取发现的文件
        if self.step <= 4:
            for path in env.fs.files:
                if path not in self.read_contents:
                    self.read_contents[path] = True
                    return f"read_file {path}"

        # 最终：提交答案（尝试从观测中提取）
        return f"done {env.task.expected_answer}"  # 作弊式答案，用于演示


class TrainedAgent:
    """
    模拟 "训练后" 的 Agent：高效完成任务。
    实际中这应该是经过 GRPO 训练的 LLM。
    """

    def __init__(self):
        self.name = "TrainedAgent"
        self.step = 0

    def reset(self):
        self.step = 0

    def act(self, observation: str, env: AgentEnvironment) -> str:
        """模拟训练后的高效决策。"""
        self.step += 1
        task = env.task

        # 使用预定义的最优策略（模拟训练后学到的策略）
        optimal_plans = {
            "error": ["search error", f"done {task.expected_answer}"],
            "总行数": ["list_dir /project", "read_file /project/main.py",
                       "read_file /project/utils.py", f"done {task.expected_answer}"],
            "端口号": ["search port", f"done {task.expected_answer}"],
            "总和": ["read_file /src/data.csv", f"done {task.expected_answer}"],
            "WARNING": ["search WARNING", f"done {task.expected_answer}"],
        }

        for keyword, plan in optimal_plans.items():
            if keyword in task.description:
                if self.step <= len(plan):
                    return plan[self.step - 1]

        return f"done {task.expected_answer}"


# ======================================================================
# GRPO 模拟训练
# ======================================================================

def simulate_grpo_trajectory(agent, task: Task, enable_process_reward: bool) -> Dict[str, Any]:
    """
    运行一条完整的 Agent 轨迹。

    返回：
    {
        "actions": [...],
        "observations": [...],
        "rewards": [...],
        "total_reward": float,
        "steps": int,
        "success": bool,
    }
    """
    env = AgentEnvironment(task, enable_process_reward=enable_process_reward)
    obs = env.reset()

    if hasattr(agent, "reset"):
        agent.reset()

    trajectory = {
        "actions": [],
        "observations": [obs],
        "rewards": [],
        "steps": 0,
        "success": False,
    }

    for _ in range(task.max_steps):
        action = agent.act(obs, env)
        obs, reward, done, info = env.step(action)

        trajectory["actions"].append(action)
        trajectory["observations"].append(obs)
        trajectory["rewards"].append(reward)
        trajectory["steps"] += 1

        if done:
            break

    trajectory["total_reward"] = sum(trajectory["rewards"])
    trajectory["success"] = trajectory["total_reward"] > 0.5

    return trajectory


def simulate_grpo_training(tracker, num_epochs=10, group_size=4):
    """
    模拟 GRPO 训练过程。

    在真实场景中：
    1. 用 LLM 对每个任务采样 G 条轨迹
    2. 计算每条轨迹的奖励
    3. 组内归一化得到优势
    4. 用策略梯度更新 LLM

    这里用不同能力水平的 Agent 模拟训练过程。
    """
    print("\n" + "=" * 60)
    print("🏋️ GRPO Agent 训练模拟")
    print("=" * 60)

    tasks = create_tasks()
    np.random.seed(42)

    # 模拟训练过程：Agent 从随机逐渐变为启发式再到最优
    training_metrics = {
        "epoch": [],
        "mean_reward": [],
        "success_rate": [],
        "mean_steps": [],
        "process_reward": [],
        "outcome_reward": [],
    }

    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_success = []
        epoch_steps = []
        epoch_process = []
        epoch_outcome = []

        # 模拟能力提升：随着训练进行，Agent 从随机→启发式→最优
        progress = epoch / num_epochs
        if progress < 0.3:
            agents = [RandomAgent() for _ in range(group_size)]
        elif progress < 0.7:
            agents = [HeuristicAgent() if np.random.random() < progress else RandomAgent()
                      for _ in range(group_size)]
        else:
            agents = [TrainedAgent() if np.random.random() < progress else HeuristicAgent()
                      for _ in range(group_size)]

        for task in tasks:
            # GRPO 核心：对同一任务采样 G 条轨迹
            group_trajectories = []
            for agent in agents:
                traj = simulate_grpo_trajectory(agent, task, enable_process_reward=True)
                group_trajectories.append(traj)

            # 组内归一化优势
            group_rewards = [t["total_reward"] for t in group_trajectories]
            mean_r = np.mean(group_rewards)
            std_r = np.std(group_rewards) + 1e-8
            advantages = [(r - mean_r) / std_r for r in group_rewards]

            # 记录指标
            for traj, adv in zip(group_trajectories, advantages):
                epoch_rewards.append(traj["total_reward"])
                epoch_success.append(1.0 if traj["success"] else 0.0)
                epoch_steps.append(traj["steps"])

                # 分离过程奖励和结果奖励
                if traj["rewards"]:
                    process_r = sum(r for r in traj["rewards"][:-1]) if len(traj["rewards"]) > 1 else 0
                    outcome_r = traj["rewards"][-1] if traj["rewards"] else 0
                    epoch_process.append(process_r)
                    epoch_outcome.append(outcome_r)

        # 汇总
        mean_reward = np.mean(epoch_rewards)
        success_rate = np.mean(epoch_success)
        mean_steps = np.mean(epoch_steps)
        mean_process = np.mean(epoch_process) if epoch_process else 0
        mean_outcome = np.mean(epoch_outcome) if epoch_outcome else 0

        training_metrics["epoch"].append(epoch)
        training_metrics["mean_reward"].append(mean_reward)
        training_metrics["success_rate"].append(success_rate)
        training_metrics["mean_steps"].append(mean_steps)
        training_metrics["process_reward"].append(mean_process)
        training_metrics["outcome_reward"].append(mean_outcome)

        tracker.log_metric("mean_reward", mean_reward, step=epoch)
        tracker.log_metric("success_rate", success_rate, step=epoch)
        tracker.log_metric("mean_steps", mean_steps, step=epoch)
        tracker.log_metric("process_reward", mean_process, step=epoch)
        tracker.log_metric("outcome_reward", mean_outcome, step=epoch)

        print(f"  Epoch {epoch+1:2d}/{num_epochs} | "
              f"奖励: {mean_reward:.3f} | "
              f"成功率: {success_rate:.1%} | "
              f"平均步数: {mean_steps:.1f} | "
              f"过程奖励: {mean_process:.3f} | "
              f"结果奖励: {mean_outcome:.3f}")

    return training_metrics


def compare_process_vs_outcome(tracker):
    """
    对比过程奖励 vs 结果奖励的训练效果。
    """
    print("\n" + "=" * 60)
    print("📊 过程奖励 vs 结果奖励 对比实验")
    print("=" * 60)

    tasks = create_tasks()
    np.random.seed(42)

    results = {"process": [], "outcome": []}

    for reward_type in ["process", "outcome"]:
        enable_process = (reward_type == "process")
        print(f"\n  {'启用过程奖励' if enable_process else '仅结果奖励'}:")

        for task in tasks[:3]:
            agent = HeuristicAgent()
            traj = simulate_grpo_trajectory(agent, task, enable_process_reward=enable_process)
            results[reward_type].append(traj["total_reward"])
            print(f"    任务: {task.description[:30]}... | 奖励: {traj['total_reward']:.3f}")

    # 记录对比结果
    tracker.log_metric("process_reward_mean", float(np.mean(results["process"])))
    tracker.log_metric("outcome_reward_mean", float(np.mean(results["outcome"])))
    tracker.log_text("reward_comparison",
                     f"过程奖励均值: {np.mean(results['process']):.3f}, "
                     f"结果奖励均值: {np.mean(results['outcome']):.3f}")

    return results


def save_agent_behavior_samples(tracker):
    """保存不同训练阶段的 Agent 行为样本。"""
    print("\n" + "=" * 60)
    print("📝 Agent 行为样本")
    print("=" * 60)

    tasks = create_tasks()
    task = tasks[0]  # 使用第一个任务

    agents = [
        ("随机Agent (训练前)", RandomAgent()),
        ("启发式Agent (训练中)", HeuristicAgent()),
        ("最优Agent (训练后)", TrainedAgent()),
    ]

    for name, agent in agents:
        print(f"\n  === {name} ===")
        traj = simulate_grpo_trajectory(agent, task, enable_process_reward=True)

        for i, (action, obs) in enumerate(zip(traj["actions"], traj["observations"][1:])):
            print(f"    步骤 {i+1}: {action}")
            print(f"    观测: {obs[:80]}...")
            print(f"    奖励: {traj['rewards'][i]:+.3f}")

        print(f"    总奖励: {traj['total_reward']:.3f} | 步数: {traj['steps']}")

        tracker.log_text(f"behavior_{name}",
                        json.dumps({"actions": traj["actions"],
                                    "total_reward": traj["total_reward"],
                                    "steps": traj["steps"]},
                                   ensure_ascii=False))


def visualize_training(metrics, comparison_results, tracker):
    """可视化训练结果。"""
    print("\n📊 生成可视化...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图 1: 训练奖励曲线
    ax = axes[0, 0]
    ax.plot(metrics["epoch"], metrics["mean_reward"], "b-o", markersize=4, label="平均奖励")
    ax.fill_between(metrics["epoch"],
                     [r - 0.1 for r in metrics["mean_reward"]],
                     [r + 0.1 for r in metrics["mean_reward"]],
                     alpha=0.2, color="blue")
    ax.set_title("GRPO Agent 训练奖励曲线")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("平均奖励")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图 2: 成功率和步骤效率
    ax = axes[0, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(metrics["epoch"], metrics["success_rate"], "g-o", markersize=4, label="成功率")
    l2 = ax2.plot(metrics["epoch"], metrics["mean_steps"], "r-s", markersize=4, label="平均步数")
    ax.set_title("成功率 & 步骤效率")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("成功率", color="green")
    ax2.set_ylabel("平均步数", color="red")
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax.grid(True, alpha=0.3)

    # 图 3: 过程奖励 vs 结果奖励
    ax = axes[1, 0]
    ax.plot(metrics["epoch"], metrics["process_reward"], "c-o", markersize=4, label="过程奖励")
    ax.plot(metrics["epoch"], metrics["outcome_reward"], "m-s", markersize=4, label="结果奖励")
    ax.set_title("过程奖励 vs 结果奖励")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("奖励")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图 4: 奖励类型对比柱状图
    ax = axes[1, 1]
    if comparison_results:
        x_labels = [f"任务{i+1}" for i in range(len(comparison_results["process"]))]
        x_pos = np.arange(len(x_labels))
        width = 0.35
        ax.bar(x_pos - width / 2, comparison_results["process"], width,
               label="过程+结果奖励", color="#4ECDC4", alpha=0.8)
        ax.bar(x_pos + width / 2, comparison_results["outcome"], width,
               label="仅结果奖励", color="#FF6B6B", alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_title("过程奖励 vs 结果奖励 效果对比")
        ax.set_ylabel("总奖励")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = PROJECT_ROOT / "day11_12_agent_rl" / "agent_rl_training.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 训练可视化已保存: {save_path}")
    tracker.log_text("training_visualization", str(save_path))


def main():
    print("=" * 60)
    print("Day 11-12 - AgentRL 训练：用 GRPO 训练文件操作 Agent")
    print("=" * 60)

    tracker = ExperimentTracker(
        experiment_name="day11_12_agent_rl_training",
        tags={"day": "11-12", "task": "agent_rl", "method": "GRPO"},
    )

    # 1. GRPO 模拟训练
    training_metrics = simulate_grpo_training(tracker, num_epochs=10, group_size=4)

    # 2. 过程奖励 vs 结果奖励对比
    comparison_results = compare_process_vs_outcome(tracker)

    # 3. 保存 Agent 行为样本
    save_agent_behavior_samples(tracker)

    # 4. 可视化
    visualize_training(training_metrics, comparison_results, tracker)

    # 5. 摘要
    print("\n" + "=" * 60)
    print("📋 实验摘要")
    print("=" * 60)
    summary = tracker.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n✅ AgentRL 训练完成！结果保存在 day11_12_agent_rl/ 目录下。")


if __name__ == "__main__":
    main()
