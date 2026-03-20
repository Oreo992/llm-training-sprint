"""
模型评估脚本
============

加载各阶段模型检查点（Base / SFT / DPO / GRPO），
在标准化测试任务上对比每个阶段的模型能力。

评估维度：
1. 任务完成率：模型是否成功完成了给定任务
2. 回复质量评分：回复的完整性、准确性、可读性
3. 工具调用准确率：是否正确选择工具和参数
4. 推理步骤效率：完成任务使用的步骤数是否合理
"""

import json
import os
import sys
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch

# 处理 import 路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.experiment_tracker import ExperimentTracker

# ============================================================================
# 测试任务定义
# ============================================================================

TEST_TASKS = [
    {
        "id": "test_001",
        "query": "帮我搜索一下大语言模型的最新进展，并总结要点。",
        "expected_tools": ["web_search"],
        "expected_keywords": ["搜索", "进展", "总结"],
        "category": "search_and_summarize",
    },
    {
        "id": "test_002",
        "query": "请计算 (15 + 27) * 3 - 18 / 6，并解释计算过程。",
        "expected_tools": ["calculator"],
        "expected_keywords": ["计算", "结果"],
        "category": "math",
    },
    {
        "id": "test_003",
        "query": "读取 report.txt 的内容，然后将摘要写入 summary.txt。",
        "expected_tools": ["file_read", "file_write"],
        "expected_keywords": ["读取", "摘要", "写入"],
        "category": "file_operation",
    },
    {
        "id": "test_004",
        "query": "查看北京的天气，帮我规划明天的出行方案。",
        "expected_tools": ["get_weather"],
        "expected_keywords": ["天气", "出行", "建议"],
        "category": "weather",
    },
    {
        "id": "test_005",
        "query": "搜索中国GDP增长率相关信息，计算关键数据，并将报告保存到 report.txt。",
        "expected_tools": ["web_search", "calculator", "file_write"],
        "expected_keywords": ["搜索", "计算", "报告"],
        "category": "multi_step",
    },
    {
        "id": "test_006",
        "query": "帮我计算 2 的 10 次方加上 3 的 5 次方。",
        "expected_tools": ["calculator"],
        "expected_keywords": ["计算", "结果"],
        "category": "math",
    },
    {
        "id": "test_007",
        "query": "查看上海的天气信息。",
        "expected_tools": ["get_weather"],
        "expected_keywords": ["天气", "上海"],
        "category": "weather",
    },
    {
        "id": "test_008",
        "query": "搜索量子计算的最新突破，写一份简要报告。",
        "expected_tools": ["web_search"],
        "expected_keywords": ["量子", "搜索", "报告"],
        "category": "search_and_summarize",
    },
]


# ============================================================================
# 评估指标计算
# ============================================================================

@dataclass
class TaskResult:
    """单个任务的评估结果"""
    task_id: str
    category: str
    response: str
    task_completed: bool          # 任务是否完成
    quality_score: float          # 回复质量 [0, 1]
    tool_accuracy: float          # 工具调用准确率 [0, 1]
    reasoning_efficiency: float   # 推理步骤效率 [0, 1]
    generation_time: float        # 生成耗时（秒）


def evaluate_tool_accuracy(response: str, expected_tools: List[str]) -> float:
    """
    评估工具调用准确率。

    检查模型是否：
    1. 使用了工具调用格式（<tool_call>）
    2. 调用了正确的工具
    3. 没有调用不必要的工具
    """
    # 提取 response 中所有工具调用
    tool_calls = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)

    if not expected_tools:
        # 如果不需要工具，没有调用就是正确的
        return 1.0 if not tool_calls else 0.5

    if not tool_calls:
        return 0.0  # 需要调用工具但没有调用

    # 解析调用的工具名
    called_tools = []
    for tc_json in tool_calls:
        try:
            tc = json.loads(tc_json)
            called_tools.append(tc.get("name", ""))
        except json.JSONDecodeError:
            continue

    if not called_tools:
        return 0.1  # 有调用格式但解析失败

    # 计算工具匹配度
    expected_set = set(expected_tools)
    called_set = set(called_tools)

    # 正确调用的工具数 / 期望调用的工具数
    correct = len(expected_set & called_set)
    precision = correct / len(called_set) if called_set else 0
    recall = correct / len(expected_set) if expected_set else 0

    # F1 score
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def evaluate_quality(response: str, expected_keywords: List[str]) -> float:
    """
    评估回复质量。

    综合考虑：
    - 关键词覆盖率：回复是否包含期望的关键词
    - 长度合理性：太短（<30字）扣分，适中（50-500字）最佳
    - 结构化程度：有序号/换行/标题等结构
    - 无幻觉标记：不包含明显的编造标记
    """
    score = 0.0

    # 关键词覆盖率（40% 权重）
    if expected_keywords:
        covered = sum(1 for kw in expected_keywords if kw in response)
        keyword_score = covered / len(expected_keywords)
        score += 0.4 * keyword_score

    # 长度合理性（20% 权重）
    length = len(response)
    if length < 10:
        length_score = 0.0
    elif length < 30:
        length_score = 0.3
    elif length < 500:
        length_score = 1.0
    elif length < 1000:
        length_score = 0.7
    else:
        length_score = 0.5
    score += 0.2 * length_score

    # 结构化程度（20% 权重）
    structure_score = 0.0
    if "\n" in response:
        structure_score += 0.3
    if any(marker in response for marker in ["1.", "2.", "3.", "- ", "* "]):
        structure_score += 0.4
    if any(marker in response for marker in ["##", "：", "："]):
        structure_score += 0.3
    score += 0.2 * min(1.0, structure_score)

    # 无幻觉（20% 权重）
    hallucination_markers = ["500%", "完全取代", "我无法", "作为AI"]
    has_hallucination = any(m in response for m in hallucination_markers)
    score += 0.2 * (0.0 if has_hallucination else 1.0)

    return min(1.0, score)


def evaluate_reasoning_efficiency(response: str, expected_tools: List[str]) -> float:
    """
    评估推理步骤效率。

    理想情况：
    - 工具调用次数与期望一致（不多也不少）
    - 回复不包含冗余的重复内容
    """
    # 统计工具调用次数
    tool_calls = re.findall(r'<tool_call>', response)
    num_calls = len(tool_calls)
    expected_calls = len(expected_tools)

    if expected_calls == 0:
        return 1.0 if num_calls == 0 else max(0.0, 1.0 - num_calls * 0.2)

    # 调用次数越接近期望值越好
    ratio = num_calls / expected_calls if expected_calls > 0 else 0
    if 0.8 <= ratio <= 1.2:  # 接近期望值
        efficiency = 1.0
    elif ratio < 0.8:  # 调用太少
        efficiency = ratio
    else:  # 调用太多
        efficiency = max(0.0, 2.0 - ratio)

    return efficiency


def evaluate_task_completion(response: str, task: Dict) -> bool:
    """
    判断任务是否完成。

    简化判断标准：
    - 回复长度 > 20 字
    - 包含至少一个期望关键词
    - 使用了至少一个正确的工具（如果需要工具的话）
    """
    if len(response) < 20:
        return False

    # 关键词检查
    has_keyword = any(kw in response for kw in task["expected_keywords"])

    # 工具检查
    if task["expected_tools"]:
        has_tool = any(tool in response for tool in task["expected_tools"])
        return has_keyword and has_tool

    return has_keyword


# ============================================================================
# 模型评估器
# ============================================================================

class ModelEvaluator:
    """
    模型评估器：加载模型并在测试任务上评估。
    """
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: Dict[str, List[TaskResult]] = {}  # stage_name -> results

    def evaluate_model(
        self,
        model,
        tokenizer,
        stage_name: str,
        tasks: List[Dict] = None,
        max_new_tokens: int = 300,
    ) -> List[TaskResult]:
        """
        在测试任务上评估模型。

        Args:
            model: 要评估的模型
            tokenizer: 对应的 tokenizer
            stage_name: 阶段名称（如 "base", "sft", "dpo"）
            tasks: 测试任务列表，默认使用 TEST_TASKS
            max_new_tokens: 最大生成 token 数
        """
        if tasks is None:
            tasks = TEST_TASKS

        print(f"\n{'=' * 60}")
        print(f"评估阶段: {stage_name}")
        print(f"{'=' * 60}")

        device = next(model.parameters()).device
        model.eval()
        results = []

        for task in tasks:
            print(f"\n  任务 {task['id']}: {task['query'][:40]}...")

            # 构造 prompt
            prompt = f"<|system|>\n你是一个有用的AI助手，可以使用工具来帮助用户完成任务。\n<|user|>\n{task['query']}\n<|assistant|>\n"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            # 生成回复
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # 评估时用 greedy 确保可复现
                    pad_token_id=tokenizer.eos_token_id,
                )
            generation_time = time.time() - start_time

            response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

            # 计算各项指标
            task_completed = evaluate_task_completion(response, task)
            quality_score = evaluate_quality(response, task["expected_keywords"])
            tool_accuracy = evaluate_tool_accuracy(response, task["expected_tools"])
            reasoning_efficiency = evaluate_reasoning_efficiency(response, task["expected_tools"])

            result = TaskResult(
                task_id=task["id"],
                category=task["category"],
                response=response,
                task_completed=task_completed,
                quality_score=quality_score,
                tool_accuracy=tool_accuracy,
                reasoning_efficiency=reasoning_efficiency,
                generation_time=generation_time,
            )
            results.append(result)

            print(f"    完成: {'是' if task_completed else '否'} | "
                  f"质量: {quality_score:.2f} | "
                  f"工具准确率: {tool_accuracy:.2f} | "
                  f"效率: {reasoning_efficiency:.2f} | "
                  f"耗时: {generation_time:.2f}s")

        self.results[stage_name] = results
        return results

    def generate_report(self) -> Dict[str, Any]:
        """
        生成综合评估报告。

        报告内容：
        - 各阶段的整体指标均值
        - 按任务类别的细分指标
        - 各阶段之间的对比（delta 变化）
        """
        report = {
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_tasks": len(TEST_TASKS),
            "stages": {},
            "comparison": {},
        }

        # 计算各阶段指标
        for stage_name, results in self.results.items():
            n = len(results)
            if n == 0:
                continue

            stage_metrics = {
                "task_completion_rate": sum(r.task_completed for r in results) / n,
                "avg_quality_score": sum(r.quality_score for r in results) / n,
                "avg_tool_accuracy": sum(r.tool_accuracy for r in results) / n,
                "avg_reasoning_efficiency": sum(r.reasoning_efficiency for r in results) / n,
                "avg_generation_time": sum(r.generation_time for r in results) / n,
            }

            # 按类别细分
            categories = set(r.category for r in results)
            category_metrics = {}
            for cat in categories:
                cat_results = [r for r in results if r.category == cat]
                cn = len(cat_results)
                category_metrics[cat] = {
                    "task_completion_rate": sum(r.task_completed for r in cat_results) / cn,
                    "avg_quality_score": sum(r.quality_score for r in cat_results) / cn,
                    "avg_tool_accuracy": sum(r.tool_accuracy for r in cat_results) / cn,
                    "count": cn,
                }

            stage_metrics["by_category"] = category_metrics
            report["stages"][stage_name] = stage_metrics

        # 阶段间对比
        stage_names = list(self.results.keys())
        for i in range(1, len(stage_names)):
            prev = stage_names[i - 1]
            curr = stage_names[i]
            if prev in report["stages"] and curr in report["stages"]:
                comparison = {}
                for metric in ["task_completion_rate", "avg_quality_score", "avg_tool_accuracy", "avg_reasoning_efficiency"]:
                    prev_val = report["stages"][prev][metric]
                    curr_val = report["stages"][curr][metric]
                    comparison[metric] = {
                        "previous": prev_val,
                        "current": curr_val,
                        "delta": curr_val - prev_val,
                        "improvement": f"{(curr_val - prev_val) * 100:+.1f}%",
                    }
                report["comparison"][f"{prev}_vs_{curr}"] = comparison

        return report

    def save_report(self, report: Dict = None) -> str:
        """保存报告为 JSON 文件"""
        if report is None:
            report = self.generate_report()

        report_path = os.path.join(self.output_dir, "evaluation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n评估报告已保存到: {report_path}")
        return report_path

    def visualize_results(self, report: Dict = None) -> str:
        """
        生成可视化图表。

        包含：
        1. 各阶段整体指标对比（雷达图/柱状图）
        2. 按任务类别的性能分解
        3. 训练进度趋势图
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib 未安装，跳过可视化。")
            return ""

        if report is None:
            report = self.generate_report()

        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Agent 模型各阶段评估报告", fontsize=16, fontweight="bold")

        stages = list(report["stages"].keys())
        if not stages:
            print("没有评估数据可以可视化。")
            return ""

        # ------------------------------------------------------------------
        # 图1：各阶段整体指标柱状图
        # ------------------------------------------------------------------
        ax1 = axes[0, 0]
        metrics = ["task_completion_rate", "avg_quality_score", "avg_tool_accuracy", "avg_reasoning_efficiency"]
        metric_labels = ["任务完成率", "回复质量", "工具准确率", "推理效率"]
        x = np.arange(len(metrics))
        width = 0.8 / len(stages)

        colors = ["#4ECDC4", "#FF6B6B", "#45B7D1", "#96CEB4"]
        for i, stage in enumerate(stages):
            values = [report["stages"][stage][m] for m in metrics]
            ax1.bar(x + i * width, values, width, label=stage, color=colors[i % len(colors)], alpha=0.85)

        ax1.set_xlabel("评估指标")
        ax1.set_ylabel("分数")
        ax1.set_title("各阶段整体指标对比")
        ax1.set_xticks(x + width * (len(stages) - 1) / 2)
        ax1.set_xticklabels(metric_labels)
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # ------------------------------------------------------------------
        # 图2：任务完成率随阶段的变化
        # ------------------------------------------------------------------
        ax2 = axes[0, 1]
        completion_rates = [report["stages"][s]["task_completion_rate"] for s in stages]
        quality_scores = [report["stages"][s]["avg_quality_score"] for s in stages]

        ax2.plot(stages, completion_rates, "o-", color="#FF6B6B", linewidth=2, markersize=8, label="任务完成率")
        ax2.plot(stages, quality_scores, "s-", color="#4ECDC4", linewidth=2, markersize=8, label="回复质量")
        ax2.set_xlabel("训练阶段")
        ax2.set_ylabel("分数")
        ax2.set_title("训练阶段性能趋势")
        ax2.set_ylim(0, 1.1)
        ax2.legend()
        ax2.grid(alpha=0.3)

        # ------------------------------------------------------------------
        # 图3：按任务类别分解（最后一个阶段）
        # ------------------------------------------------------------------
        ax3 = axes[1, 0]
        last_stage = stages[-1]
        categories = report["stages"][last_stage].get("by_category", {})
        if categories:
            cat_names = list(categories.keys())
            cat_completion = [categories[c]["task_completion_rate"] for c in cat_names]
            cat_quality = [categories[c]["avg_quality_score"] for c in cat_names]

            x_cat = np.arange(len(cat_names))
            ax3.bar(x_cat - 0.15, cat_completion, 0.3, label="完成率", color="#FF6B6B", alpha=0.85)
            ax3.bar(x_cat + 0.15, cat_quality, 0.3, label="质量", color="#4ECDC4", alpha=0.85)
            ax3.set_xlabel("任务类别")
            ax3.set_ylabel("分数")
            ax3.set_title(f"按类别分解 ({last_stage})")
            ax3.set_xticks(x_cat)
            ax3.set_xticklabels(cat_names, rotation=30, ha="right")
            ax3.set_ylim(0, 1.1)
            ax3.legend()
            ax3.grid(axis="y", alpha=0.3)

        # ------------------------------------------------------------------
        # 图4：阶段间改进幅度
        # ------------------------------------------------------------------
        ax4 = axes[1, 1]
        if report["comparison"]:
            comp_name = list(report["comparison"].keys())[0]
            comp = report["comparison"][comp_name]
            delta_names = list(comp.keys())
            delta_values = [comp[m]["delta"] for m in delta_names]
            delta_labels = ["完成率", "质量", "工具准确率", "推理效率"]

            bar_colors = ["#4ECDC4" if v >= 0 else "#FF6B6B" for v in delta_values]
            ax4.barh(delta_labels, delta_values, color=bar_colors, alpha=0.85)
            ax4.set_xlabel("变化幅度")
            ax4.set_title(f"改进幅度 ({comp_name})")
            ax4.axvline(x=0, color="black", linewidth=0.5)
            ax4.grid(axis="x", alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "需要至少两个阶段\n才能显示对比", ha="center", va="center", fontsize=12)
            ax4.set_title("阶段间改进幅度")

        plt.tight_layout()
        chart_path = os.path.join(self.output_dir, "evaluation_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"评估图表已保存到: {chart_path}")
        return chart_path


# ============================================================================
# 快速评估（不需要加载真实模型，使用模拟数据演示评估流程）
# ============================================================================

def run_demo_evaluation():
    """
    演示评估流程（使用模拟的模型输出）。

    当没有 GPU 或未训练模型时，可以用此函数了解评估的完整流程。
    """
    print("=" * 60)
    print("演示模式：使用模拟输出进行评估")
    print("=" * 60)

    evaluator = ModelEvaluator(
        output_dir=os.path.join(os.path.dirname(__file__), "evaluation_results")
    )

    # 模拟各阶段的输出
    stage_outputs = {
        "base": {
            # 基座模型：通常不会正确使用工具，只会生成自然语言
            "default": "这个问题很有趣。让我想想...\n我觉得答案可能是这样的。",
        },
        "sft": {
            # SFT 后：学会了工具调用格式，但可能不够精准
            "search_and_summarize": '<tool_call>{"name": "web_search", "arguments": {"query": "大语言模型最新进展"}}</tool_call>\n\n根据搜索结果，以下是最新进展：\n1. 模型能力持续提升\n2. 多模态能力增强',
            "math": '<tool_call>{"name": "calculator", "arguments": {"expression": "(15+27)*3-18/6"}}</tool_call>\n\n计算结果为123。',
            "file_operation": '<tool_call>{"name": "file_read", "arguments": {"path": "report.txt"}}</tool_call>\n\n<tool_call>{"name": "file_write", "arguments": {"path": "summary.txt", "content": "摘要内容"}}</tool_call>\n\n已完成文件读取和摘要写入。',
            "weather": '<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>\n\n天气晴好，建议外出。',
            "multi_step": '<tool_call>{"name": "web_search", "arguments": {"query": "GDP"}}</tool_call>\n\n搜索完成，报告已生成。',
        },
        "dpo": {
            # DPO 后：回复更完整、更有条理，工具使用更合理
            "search_and_summarize": '<tool_call>{"name": "web_search", "arguments": {"query": "大语言模型 最新进展 2024", "max_results": 5}}</tool_call>\n\n根据搜索结果，以下是大语言模型的最新进展要点：\n\n1. **技术突破**：新一代模型在推理能力上取得显著提升\n2. **应用落地**：越来越多企业将大模型应用到实际场景\n3. **开源生态**：开源模型性能逐步接近闭源模型\n\n总结：大语言模型领域正在快速发展，值得持续关注。',
            "math": '<tool_call>{"name": "calculator", "arguments": {"expression": "(15+27)*3-18/6"}}</tool_call>\n\n计算结果为 **123.0**。\n\n计算过程：\n1. 先计算括号：15 + 27 = 42\n2. 乘法：42 × 3 = 126\n3. 除法：18 ÷ 6 = 3\n4. 减法：126 - 3 = 123',
            "file_operation": '<tool_call>{"name": "file_read", "arguments": {"path": "report.txt"}}</tool_call>\n\n<tool_call>{"name": "file_write", "arguments": {"path": "summary.txt", "content": "报告摘要：文件包含关键数据分析结果，核心要点已提取。"}}</tool_call>\n\n已完成操作：\n1. 读取了 report.txt 的内容\n2. 提取关键信息并生成摘要\n3. 将摘要写入 summary.txt',
            "weather": '<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>\n\n北京天气：晴，气温12°C，北风3级。\n\n出行建议：\n- 穿着：建议穿外套，早晚温差较大\n- 交通：天气晴好，适合户外活动\n- 注意：湿度较低，注意补水',
            "multi_step": '<tool_call>{"name": "web_search", "arguments": {"query": "中国GDP增长率 2024"}}</tool_call>\n\n<tool_call>{"name": "calculator", "arguments": {"expression": "100 * 1.05 ** 5"}}</tool_call>\n\n<tool_call>{"name": "file_write", "arguments": {"path": "report.txt", "content": "GDP增长率研究报告"}}</tool_call>\n\n已完成研究报告：\n1. 搜索了最新GDP相关信息\n2. 计算了关键增长数据\n3. 将完整报告保存到 report.txt',
        },
    }

    # 对每个阶段进行评估
    for stage_name, outputs in stage_outputs.items():
        results = []
        print(f"\n{'=' * 60}")
        print(f"评估阶段: {stage_name}")
        print(f"{'=' * 60}")

        for task in TEST_TASKS:
            # 获取对应的模拟输出
            response = outputs.get(task["category"], outputs.get("default", ""))

            task_completed = evaluate_task_completion(response, task)
            quality_score = evaluate_quality(response, task["expected_keywords"])
            tool_accuracy = evaluate_tool_accuracy(response, task["expected_tools"])
            reasoning_efficiency = evaluate_reasoning_efficiency(response, task["expected_tools"])

            result = TaskResult(
                task_id=task["id"],
                category=task["category"],
                response=response,
                task_completed=task_completed,
                quality_score=quality_score,
                tool_accuracy=tool_accuracy,
                reasoning_efficiency=reasoning_efficiency,
                generation_time=0.5,  # 模拟
            )
            results.append(result)

            print(f"  {task['id']}: 完成={'是' if task_completed else '否'} | "
                  f"质量={quality_score:.2f} | "
                  f"工具={tool_accuracy:.2f} | "
                  f"效率={reasoning_efficiency:.2f}")

        evaluator.results[stage_name] = results

    # 生成报告
    report = evaluator.generate_report()
    evaluator.save_report(report)
    chart_path = evaluator.visualize_results(report)

    # 打印摘要
    print("\n" + "=" * 60)
    print("评估摘要")
    print("=" * 60)
    for stage, metrics in report["stages"].items():
        print(f"\n  [{stage}]")
        print(f"    任务完成率: {metrics['task_completion_rate']:.2%}")
        print(f"    回复质量:   {metrics['avg_quality_score']:.2f}")
        print(f"    工具准确率: {metrics['avg_tool_accuracy']:.2f}")
        print(f"    推理效率:   {metrics['avg_reasoning_efficiency']:.2f}")

    if report["comparison"]:
        print("\n  [阶段对比]")
        for comp_name, comp in report["comparison"].items():
            print(f"    {comp_name}:")
            for metric, data in comp.items():
                print(f"      {metric}: {data['improvement']}")

    return report


# ============================================================================
# 完整评估（加载真实模型检查点）
# ============================================================================

def run_full_evaluation(checkpoint_dir: str = "checkpoints"):
    """
    加载各阶段的真实模型检查点进行评估。

    需要先运行 end_to_end_agent_training.py 生成检查点。
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers 未安装，切换到演示模式。")
        return run_demo_evaluation()

    evaluator = ModelEvaluator(
        output_dir=os.path.join(os.path.dirname(__file__), "evaluation_results")
    )

    stages = {
        "base": "Qwen/Qwen2-0.5B",
        "sft": os.path.join(checkpoint_dir, "sft"),
        "dpo": os.path.join(checkpoint_dir, "dpo"),
    }

    # 如果有 GRPO 检查点也加载
    grpo_path = os.path.join(checkpoint_dir, "grpo")
    if os.path.exists(grpo_path):
        stages["grpo"] = grpo_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for stage_name, model_path in stages.items():
        if not os.path.exists(model_path) and stage_name != "base":
            print(f"跳过 {stage_name}：检查点 {model_path} 不存在")
            continue

        print(f"\n加载模型: {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                trust_remote_code=True,
            ).to(device)

            evaluator.evaluate_model(model, tokenizer, stage_name)

            # 释放显存
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"加载 {stage_name} 失败: {e}")
            continue

    report = evaluator.generate_report()
    evaluator.save_report(report)
    evaluator.visualize_results(report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--demo", action="store_true", help="使用模拟数据进行演示评估")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="检查点目录")
    args = parser.parse_args()

    if args.demo:
        run_demo_evaluation()
    else:
        run_full_evaluation(args.checkpoint_dir)
