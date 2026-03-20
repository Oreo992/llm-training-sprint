"""
Day 11-12 - Agent 环境定义：文件操作 Agent
============================================

定义一个简单但完整的文件操作 Agent 环境，用于 RL 训练。

环境设计：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Agent 需要在一个虚拟文件系统中完成任务，例如：
    - 查找包含特定内容的文件
    - 创建或修改文件
    - 整理文件目录结构

    动作空间 (Action Space):
    ────────────────────────────────────────
    1. read_file(path)   - 读取文件内容
    2. write_file(path, content) - 写入文件
    3. list_dir(path)    - 列出目录内容
    4. search(keyword)   - 搜索包含关键词的文件
    5. done(answer)      - 提交最终答案

    状态 (State):
    ────────────────────────────────────────
    - 当前任务描述
    - 已执行的动作历史
    - 最近一次动作的返回结果
    - 剩余步数

    奖励函数 (Reward):
    ────────────────────────────────────────
    - 任务完成度: 0.0 ~ 1.0（答案是否正确）
    - 步骤效率: 奖励较少步骤完成任务
    - 过程奖励: 每个有意义的中间步骤给予小奖励
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_tracker import ExperimentTracker


# ======================================================================
# 虚拟文件系统
# ======================================================================

class VirtualFileSystem:
    """
    虚拟文件系统，模拟真实的文件操作。
    所有操作都在内存中完成，不会影响真实文件系统。
    """

    def __init__(self):
        """初始化虚拟文件系统，包含一些预设文件。"""
        self.files: Dict[str, str] = {}
        self.access_log: List[str] = []

    def add_file(self, path: str, content: str):
        """添加文件到虚拟文件系统。"""
        self.files[path] = content

    def read_file(self, path: str) -> Tuple[bool, str]:
        """读取文件内容。返回 (成功, 内容或错误信息)。"""
        self.access_log.append(f"READ: {path}")
        if path in self.files:
            return True, self.files[path]
        return False, f"Error: 文件 '{path}' 不存在"

    def write_file(self, path: str, content: str) -> Tuple[bool, str]:
        """写入文件。返回 (成功, 结果信息)。"""
        self.access_log.append(f"WRITE: {path}")
        self.files[path] = content
        return True, f"成功写入文件 '{path}' ({len(content)} 字符)"

    def list_dir(self, dir_path: str) -> Tuple[bool, str]:
        """列出目录下的文件。返回 (成功, 文件列表)。"""
        self.access_log.append(f"LIST: {dir_path}")
        # 标准化路径
        if not dir_path.endswith("/"):
            dir_path += "/"

        items = set()
        for path in self.files:
            if path.startswith(dir_path):
                # 获取相对路径的第一级
                relative = path[len(dir_path):]
                first_part = relative.split("/")[0]
                if "/" in relative:
                    items.add(first_part + "/")  # 目录
                else:
                    items.add(first_part)  # 文件
            elif dir_path == "/" or dir_path == "./":
                first_part = path.lstrip("/").split("/")[0]
                if "/" in path.lstrip("/"):
                    items.add(first_part + "/")
                else:
                    items.add(first_part)

        if items:
            return True, "\n".join(sorted(items))
        return False, f"Error: 目录 '{dir_path}' 为空或不存在"

    def search(self, keyword: str) -> Tuple[bool, str]:
        """搜索包含关键词的文件。返回 (成功, 匹配文件列表)。"""
        self.access_log.append(f"SEARCH: {keyword}")
        matches = []
        for path, content in self.files.items():
            if keyword.lower() in content.lower() or keyword.lower() in path.lower():
                # 找到匹配行
                for i, line in enumerate(content.split("\n"), 1):
                    if keyword.lower() in line.lower():
                        matches.append(f"{path}:{i}: {line.strip()}")
                        break
                else:
                    matches.append(f"{path}: (文件名匹配)")

        if matches:
            return True, "\n".join(matches[:10])  # 最多返回 10 条
        return False, f"未找到包含 '{keyword}' 的文件"


# ======================================================================
# 任务定义
# ======================================================================

@dataclass
class Task:
    """Agent 任务定义。"""
    description: str          # 任务描述
    expected_answer: str      # 期望的最终答案
    setup_files: Dict[str, str]  # 初始文件系统内容
    difficulty: str = "easy"  # easy / medium / hard
    max_steps: int = 10       # 最大步数
    hints: List[str] = field(default_factory=list)


def create_tasks() -> List[Task]:
    """创建一组文件操作任务。"""
    tasks = [
        # 任务 1: 简单文件查找
        Task(
            description="请找到 /data 目录下包含 'error' 关键词的文件名，用 done 提交答案。",
            expected_answer="log.txt",
            setup_files={
                "/data/readme.md": "# 项目说明\n这是一个示例项目。",
                "/data/log.txt": "2024-01-01 INFO: 启动成功\n2024-01-02 ERROR: 连接超时\n2024-01-03 INFO: 恢复正常",
                "/data/config.json": '{"host": "localhost", "port": 8080}',
                "/data/notes.txt": "今日待办：完成报告",
            },
            difficulty="easy",
            max_steps=5,
            hints=["先用 list_dir 查看目录", "再用 search 搜索关键词"],
        ),
        # 任务 2: 多步查找
        Task(
            description="请找出 /project 中所有 Python 文件的总行数，用 done 提交数字答案。",
            expected_answer="15",
            setup_files={
                "/project/main.py": "import os\nimport sys\n\ndef main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()\n",
                "/project/utils.py": "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n",
                "/project/readme.md": "# Project\nA simple project.",
                "/project/data/input.txt": "1 2 3 4 5",
            },
            difficulty="medium",
            max_steps=8,
            hints=["先列出目录找到 .py 文件", "读取每个 .py 文件计算行数"],
        ),
        # 任务 3: 信息提取
        Task(
            description="在 /config 目录中找到数据库端口号，用 done 提交端口号。",
            expected_answer="5432",
            setup_files={
                "/config/app.yaml": "app:\n  name: myapp\n  version: 1.0\n",
                "/config/db.yaml": "database:\n  host: db.example.com\n  port: 5432\n  name: production\n",
                "/config/redis.yaml": "redis:\n  host: redis.example.com\n  port: 6379\n",
            },
            difficulty="easy",
            max_steps=6,
            hints=["搜索 'database' 或 'port' 关键词"],
        ),
        # 任务 4: 文件创建
        Task(
            description="读取 /src/data.csv 中的数据，计算第二列的总和，将结果写入 /output/result.txt，然后用 done 提交总和。",
            expected_answer="150",
            setup_files={
                "/src/data.csv": "name,score\nAlice,30\nBob,45\nCharlie,25\nDiana,50\n",
            },
            difficulty="hard",
            max_steps=10,
            hints=["读取 CSV", "计算总和", "写入结果文件"],
        ),
        # 任务 5: 多文件分析
        Task(
            description="找出 /logs 目录中出现 'WARNING' 最多的日志文件名，用 done 提交。",
            expected_answer="app.log",
            setup_files={
                "/logs/app.log": "WARNING: 内存不足\nINFO: 正常\nWARNING: CPU 过高\nWARNING: 磁盘空间不足\nERROR: 崩溃",
                "/logs/access.log": "GET /api 200\nPOST /login 401\nWARNING: 频繁请求\nGET /data 200",
                "/logs/system.log": "INFO: 系统启动\nINFO: 服务就绪\nWARNING: 时钟不同步",
            },
            difficulty="medium",
            max_steps=10,
            hints=["列出日志文件", "逐个读取并计数 WARNING"],
        ),
    ]
    return tasks


# ======================================================================
# Agent 环境
# ======================================================================

class AgentEnvironment:
    """
    文件操作 Agent 环境。
    支持多轮交互，每轮 Agent 执行一个动作并获得观测和奖励。

    动作格式（字符串，模拟 LLM 的文本输出）：
    - "read_file /path/to/file"
    - "write_file /path/to/file content..."
    - "list_dir /path/to/dir"
    - "search keyword"
    - "done answer"
    """

    # 动作空间定义
    ACTION_SPACE = {
        "read_file": "读取指定路径的文件内容",
        "write_file": "向指定路径写入内容",
        "list_dir": "列出指定目录下的文件和子目录",
        "search": "搜索包含关键词的文件",
        "done": "提交最终答案，结束任务",
    }

    def __init__(self, task: Task, enable_process_reward: bool = True):
        """
        初始化环境。

        Args:
            task: 任务定义
            enable_process_reward: 是否启用过程奖励（中间步骤的奖励）
        """
        self.task = task
        self.enable_process_reward = enable_process_reward
        self.fs = VirtualFileSystem()
        self.step_count = 0
        self.done_flag = False
        self.history: List[Dict[str, Any]] = []
        self.submitted_answer: Optional[str] = None

        # 过程奖励追踪
        self._explored_dirs = set()
        self._read_files = set()
        self._useful_searches = 0

        # 初始化文件系统
        for path, content in task.setup_files.items():
            self.fs.add_file(path, content)

    def reset(self) -> str:
        """重置环境，返回初始观测。"""
        self.step_count = 0
        self.done_flag = False
        self.history = []
        self.submitted_answer = None
        self._explored_dirs = set()
        self._read_files = set()
        self._useful_searches = 0

        # 重新初始化文件系统
        self.fs = VirtualFileSystem()
        for path, content in self.task.setup_files.items():
            self.fs.add_file(path, content)

        return self._get_initial_observation()

    def _get_initial_observation(self) -> str:
        """返回初始观测（任务描述）。"""
        obs = f"任务: {self.task.description}\n"
        obs += f"最大步数: {self.task.max_steps}\n"
        obs += f"可用动作: {', '.join(self.ACTION_SPACE.keys())}\n"
        obs += "请开始执行任务。"
        return obs

    def step(self, action_str: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        执行一个动作。

        Args:
            action_str: Agent 输出的动作字符串

        Returns:
            (observation, reward, done, info)
            - observation: 动作执行后的观测
            - reward: 即时奖励
            - done: 是否结束
            - info: 额外信息
        """
        self.step_count += 1
        reward = 0.0
        info = {"step": self.step_count}

        # 解析动作
        action_name, action_args = self._parse_action(action_str)

        if action_name is None:
            observation = f"Error: 无法解析动作 '{action_str}'。可用动作: {', '.join(self.ACTION_SPACE.keys())}"
            reward = -0.1  # 无效动作惩罚
        elif action_name == "read_file":
            success, result = self.fs.read_file(action_args)
            observation = result
            if success and self.enable_process_reward:
                if action_args not in self._read_files:
                    self._read_files.add(action_args)
                    reward = 0.05  # 过程奖励：读取新文件
        elif action_name == "write_file":
            # 解析: write_file /path content...
            parts = action_args.split(" ", 1)
            if len(parts) == 2:
                path, content = parts
                success, result = self.fs.write_file(path, content)
                observation = result
                if success and self.enable_process_reward:
                    reward = 0.05
            else:
                observation = "Error: write_file 需要路径和内容参数"
                reward = -0.05
        elif action_name == "list_dir":
            success, result = self.fs.list_dir(action_args)
            observation = result
            if success and self.enable_process_reward:
                if action_args not in self._explored_dirs:
                    self._explored_dirs.add(action_args)
                    reward = 0.05
        elif action_name == "search":
            success, result = self.fs.search(action_args)
            observation = result
            if success and self.enable_process_reward:
                self._useful_searches += 1
                reward = 0.05
        elif action_name == "done":
            self.done_flag = True
            self.submitted_answer = action_args.strip()
            observation = f"已提交答案: {self.submitted_answer}"
            # 计算最终奖励
            reward = self._compute_final_reward()
        else:
            observation = f"Error: 未知动作 '{action_name}'"
            reward = -0.1

        # 检查是否超出步数限制
        if self.step_count >= self.task.max_steps and not self.done_flag:
            self.done_flag = True
            observation += "\n⏰ 已达到最大步数限制，任务结束。"
            reward = -0.2  # 超时惩罚

        # 记录历史
        self.history.append({
            "step": self.step_count,
            "action": action_str,
            "observation": observation,
            "reward": reward,
        })

        info["total_reward"] = sum(h["reward"] for h in self.history)
        info["action_name"] = action_name

        return observation, reward, self.done_flag, info

    def _parse_action(self, action_str: str) -> Tuple[Optional[str], str]:
        """解析动作字符串为 (动作名, 参数)。"""
        action_str = action_str.strip()
        for action_name in self.ACTION_SPACE:
            if action_str.startswith(action_name):
                args = action_str[len(action_name):].strip()
                return action_name, args
        return None, ""

    def _compute_final_reward(self) -> float:
        """
        计算最终奖励。

        奖励组成：
        1. 任务完成度 (0 ~ 1.0): 答案是否正确
        2. 步骤效率 (0 ~ 0.3): 用较少步骤完成给予额外奖励
        """
        reward = 0.0

        # 1. 任务完成度
        if self.submitted_answer:
            expected = self.task.expected_answer.strip().lower()
            submitted = self.submitted_answer.strip().lower()

            if submitted == expected:
                reward += 1.0  # 完全匹配
            elif expected in submitted or submitted in expected:
                reward += 0.5  # 部分匹配
            else:
                # 尝试数值比较
                try:
                    if abs(float(submitted) - float(expected)) < 0.01:
                        reward += 1.0
                except ValueError:
                    reward += 0.0

        # 2. 步骤效率奖励
        if reward > 0.5:  # 只有答对才给效率奖励
            efficiency = max(0, 1 - self.step_count / self.task.max_steps)
            reward += 0.3 * efficiency

        return reward

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """返回完整的交互轨迹。"""
        return self.history

    def get_total_reward(self) -> float:
        """返回累计总奖励。"""
        return sum(h["reward"] for h in self.history)

    def render(self) -> str:
        """渲染当前状态为可读字符串。"""
        lines = [f"=== 任务: {self.task.description} ==="]
        lines.append(f"步数: {self.step_count}/{self.task.max_steps}")
        if self.history:
            last = self.history[-1]
            lines.append(f"最后动作: {last['action']}")
            lines.append(f"最后观测: {last['observation'][:100]}")
        lines.append(f"累计奖励: {self.get_total_reward():.3f}")
        return "\n".join(lines)


# ======================================================================
# 演示：手动运行 Agent 环境
# ======================================================================

def demo_environment():
    """演示 Agent 环境的基本使用。"""
    print("\n" + "=" * 60)
    print("📂 Agent 环境演示")
    print("=" * 60)

    tracker = ExperimentTracker(
        experiment_name="day11_agent_env_demo",
        tags={"day": "11", "task": "agent_env"},
    )

    tasks = create_tasks()

    for task_idx, task in enumerate(tasks[:3]):  # 演示前 3 个任务
        print(f"\n{'─' * 50}")
        print(f"📋 任务 {task_idx + 1}: {task.description}")
        print(f"   难度: {task.difficulty} | 最大步数: {task.max_steps}")
        print(f"{'─' * 50}")

        env = AgentEnvironment(task)
        obs = env.reset()
        print(f"  初始观测: {obs}")

        # 模拟一个简单的规则 Agent
        actions = simulate_rule_agent(task)
        total_reward = 0

        for action in actions:
            obs, reward, done, info = env.step(action)
            total_reward = info["total_reward"]
            print(f"\n  动作: {action}")
            print(f"  观测: {obs[:120]}")
            print(f"  即时奖励: {reward:+.3f} | 累计奖励: {total_reward:.3f}")

            if done:
                break

        print(f"\n  ✅ 任务 {task_idx+1} 结束 | 总奖励: {total_reward:.3f}")
        tracker.log_metric(f"task_{task_idx}_reward", total_reward)
        tracker.log_text(f"task_{task_idx}_trajectory",
                        json.dumps(env.get_trajectory(), ensure_ascii=False, indent=2)[:500])

    # 摘要
    print("\n" + "=" * 60)
    print("📋 实验摘要")
    print("=" * 60)
    summary = tracker.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")


def simulate_rule_agent(task: Task) -> List[str]:
    """
    模拟一个基于规则的 Agent（用于演示环境功能）。
    真实训练中，这些动作应该由 LLM 生成。
    """
    if "error" in task.description.lower():
        return ["list_dir /data", "search error", "done log.txt"]
    elif "总行数" in task.description:
        return ["list_dir /project", "read_file /project/main.py",
                "read_file /project/utils.py", "done 15"]
    elif "端口号" in task.description or "数据库" in task.description:
        return ["list_dir /config", "search port", "read_file /config/db.yaml", "done 5432"]
    elif "总和" in task.description:
        return ["read_file /src/data.csv", "write_file /output/result.txt 150", "done 150"]
    elif "WARNING" in task.description:
        return ["list_dir /logs", "read_file /logs/app.log",
                "read_file /logs/access.log", "read_file /logs/system.log", "done app.log"]
    else:
        return ["list_dir /", "done unknown"]


if __name__ == "__main__":
    demo_environment()
