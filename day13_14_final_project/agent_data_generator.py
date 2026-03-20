"""
Agent 训练数据生成器
==================

为端到端 Agent 训练流程生成标准化的训练数据：
1. SFT 数据：function calling 格式的对话数据
2. DPO 数据：好/坏 Agent 轨迹的偏好对（chosen / rejected）

核心概念：
- Function Calling：模型通过结构化的 JSON 调用外部工具，而非生成自然语言
- Agent 轨迹（Trajectory）：一次完整的任务执行过程，包含多轮「思考→调用→观察」
- 偏好数据：同一个 prompt，一条好的轨迹（chosen）和一条差的轨迹（rejected）
"""

import json
import random
import os
from typing import Any, Dict, List, Tuple

# ============================================================================
# 工具定义 —— 模拟 Agent 可以使用的外部工具
# ============================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "在互联网上搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "max_results": {"type": "integer", "description": "最大返回结果数", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '2+3*4'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "读取文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "将内容写入文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
        },
    },
]

# ============================================================================
# 任务模板 —— 每个模板包含 user_query 以及对应的 good/bad 轨迹生成逻辑
# ============================================================================

TASK_TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "search_and_summarize",
        "user_query": "帮我搜索一下 {topic} 的最新进展，并总结要点。",
        "topics": ["大语言模型", "量子计算", "可控核聚变", "脑机接口", "自动驾驶"],
        "system_prompt": "你是一个有用的AI助手，可以使用工具来帮助用户完成任务。",
    },
    {
        "id": "math_problem",
        "user_query": "请计算 {expression}，并解释计算过程。",
        "expressions": ["(15 + 27) * 3 - 18 / 6", "2**10 + 3**5", "sqrt(144) + log(100)"],
        "system_prompt": "你是一个有用的AI助手，可以使用工具来帮助用户完成任务。",
    },
    {
        "id": "file_operation",
        "user_query": "读取 {filename} 的内容，然后将摘要写入 summary.txt。",
        "filenames": ["report.txt", "data.csv", "notes.md"],
        "system_prompt": "你是一个有用的AI助手，可以使用工具来帮助用户完成任务。",
    },
    {
        "id": "weather_plan",
        "user_query": "查看 {city} 的天气，帮我规划明天的出行方案。",
        "cities": ["北京", "上海", "深圳", "成都", "杭州"],
        "system_prompt": "你是一个有用的AI助手，可以使用工具来帮助用户完成任务。",
    },
    {
        "id": "multi_step_research",
        "user_query": "搜索 {topic} 相关信息，计算关键数据，并将报告保存到 report.txt。",
        "topics": ["中国GDP增长率", "全球碳排放数据", "AI行业融资趋势"],
        "system_prompt": "你是一个有用的AI助手，可以使用工具来帮助用户完成任务。",
    },
]

# ============================================================================
# 模拟工具返回结果
# ============================================================================

MOCK_SEARCH_RESULTS = {
    "大语言模型": [
        {"title": "GPT-5 发布，推理能力大幅提升", "snippet": "OpenAI发布GPT-5，在数学推理和代码生成上表现优异..."},
        {"title": "开源模型 Qwen2.5 性能逼近闭源", "snippet": "阿里发布Qwen2.5系列，在多项基准上接近GPT-4..."},
    ],
    "量子计算": [
        {"title": "Google 量子纠错取得突破", "snippet": "Willow芯片实现低于阈值的量子纠错..."},
    ],
}

MOCK_WEATHER = {
    "北京": {"temp": "12°C", "condition": "晴", "wind": "北风3级", "humidity": "35%"},
    "上海": {"temp": "18°C", "condition": "多云", "wind": "东南风2级", "humidity": "65%"},
    "深圳": {"temp": "25°C", "condition": "阴", "wind": "南风2级", "humidity": "78%"},
    "成都": {"temp": "15°C", "condition": "小雨", "wind": "微风", "humidity": "82%"},
    "杭州": {"temp": "16°C", "condition": "晴转多云", "wind": "东风2级", "humidity": "60%"},
}


def _mock_tool_result(tool_name: str, arguments: Dict) -> str:
    """模拟工具调用返回结果"""
    if tool_name == "web_search":
        query = arguments.get("query", "")
        for key, results in MOCK_SEARCH_RESULTS.items():
            if key in query:
                return json.dumps(results, ensure_ascii=False)
        return json.dumps([{"title": f"关于{query}的最新研究", "snippet": f"{query}领域近期取得重要进展..."}], ensure_ascii=False)
    elif tool_name == "calculator":
        expr = arguments.get("expression", "0")
        try:
            import math
            result = eval(expr, {"__builtins__": {}}, {"sqrt": math.sqrt, "log": math.log10, "pi": math.pi})
            return str(result)
        except Exception:
            return "计算错误"
    elif tool_name == "file_read":
        return f"[文件内容] 这是 {arguments.get('path', '')} 的模拟内容。包含一些数据和分析结果。"
    elif tool_name == "file_write":
        return f"文件 {arguments.get('path', '')} 写入成功。"
    elif tool_name == "get_weather":
        city = arguments.get("city", "北京")
        weather = MOCK_WEATHER.get(city, {"temp": "20°C", "condition": "晴", "wind": "微风", "humidity": "50%"})
        return json.dumps(weather, ensure_ascii=False)
    return "未知工具"


# ============================================================================
# 轨迹生成 —— 好轨迹 vs 坏轨迹
# ============================================================================

def _generate_good_trajectory(template: Dict, filled_query: str) -> List[Dict[str, str]]:
    """
    生成高质量 Agent 轨迹（chosen）。

    好轨迹的特征：
    - 先思考再行动（CoT）
    - 正确选择工具和参数
    - 合理利用工具返回结果
    - 给出完整、有条理的最终回答
    """
    messages = [
        {"role": "system", "content": template["system_prompt"]},
        {"role": "user", "content": filled_query},
    ]

    task_id = template["id"]

    if task_id == "search_and_summarize":
        topic = filled_query.split("搜索一下 ")[1].split(" 的")[0]
        # 步骤1：思考 + 调用搜索
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"query": f"{topic} 最新进展 2024", "max_results": 5}, ensure_ascii=False),
                },
            }],
        })
        search_result = _mock_tool_result("web_search", {"query": topic})
        messages.append({"role": "tool", "tool_call_id": "call_001", "content": search_result})
        # 步骤2：总结
        messages.append({
            "role": "assistant",
            "content": f"根据搜索结果，以下是{topic}的最新进展要点：\n\n"
                       f"1. 技术突破：该领域近期取得了重要技术突破，多项关键指标显著提升。\n"
                       f"2. 产业应用：越来越多的企业开始将{topic}技术落地到实际场景中。\n"
                       f"3. 未来趋势：专家预测{topic}将在未来几年持续快速发展。\n\n"
                       f"如需更详细的信息，我可以进一步搜索。",
        })
    elif task_id == "math_problem":
        expr = filled_query.split("请计算 ")[1].split("，")[0]
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"expression": expr}),
                },
            }],
        })
        calc_result = _mock_tool_result("calculator", {"expression": expr})
        messages.append({"role": "tool", "tool_call_id": "call_001", "content": calc_result})
        messages.append({
            "role": "assistant",
            "content": f"计算结果为 {calc_result}。\n\n计算过程：\n- 表达式：{expr}\n- 按照运算优先级，先计算括号和幂运算，再计算乘除，最后计算加减。",
        })
    elif task_id == "file_operation":
        filename = filled_query.split("读取 ")[1].split(" 的")[0]
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "file_read",
                    "arguments": json.dumps({"path": filename}),
                },
            }],
        })
        read_result = _mock_tool_result("file_read", {"path": filename})
        messages.append({"role": "tool", "tool_call_id": "call_001", "content": read_result})
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_002",
                "type": "function",
                "function": {
                    "name": "file_write",
                    "arguments": json.dumps({"path": "summary.txt", "content": f"{filename}的摘要：文件包含重要数据和分析结果，核心要点已提取。"}, ensure_ascii=False),
                },
            }],
        })
        write_result = _mock_tool_result("file_write", {"path": "summary.txt"})
        messages.append({"role": "tool", "tool_call_id": "call_002", "content": write_result})
        messages.append({
            "role": "assistant",
            "content": f"已完成操作：\n1. 读取了 {filename} 的内容\n2. 提取了关键信息并生成摘要\n3. 将摘要写入了 summary.txt",
        })
    elif task_id == "weather_plan":
        city = filled_query.split("查看 ")[1].split(" 的")[0]
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": city}, ensure_ascii=False),
                },
            }],
        })
        weather_result = _mock_tool_result("get_weather", {"city": city})
        messages.append({"role": "tool", "tool_call_id": "call_001", "content": weather_result})
        weather = json.loads(weather_result)
        messages.append({
            "role": "assistant",
            "content": f"{city}明天天气预报：{weather['condition']}，气温{weather['temp']}，{weather['wind']}。\n\n"
                       f"出行建议：\n"
                       f"- 穿着：根据{weather['temp']}的气温，建议适当增减衣物。\n"
                       f"- 出行方式：{'建议携带雨具' if '雨' in weather['condition'] else '适合户外活动'}。\n"
                       f"- 注意事项：湿度{weather['humidity']}，注意补水。",
        })
    elif task_id == "multi_step_research":
        topic = filled_query.split("搜索 ")[1].split(" 相关")[0]
        # 搜索
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"query": topic, "max_results": 5}, ensure_ascii=False),
                },
            }],
        })
        search_result = _mock_tool_result("web_search", {"query": topic})
        messages.append({"role": "tool", "tool_call_id": "call_001", "content": search_result})
        # 计算
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_002",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"expression": "100 * 1.05 ** 5"}),
                },
            }],
        })
        calc_result = _mock_tool_result("calculator", {"expression": "100 * 1.05 ** 5"})
        messages.append({"role": "tool", "tool_call_id": "call_002", "content": calc_result})
        # 写报告
        report_content = f"# {topic} 研究报告\n\n## 搜索结果摘要\n基于最新数据...\n\n## 数据分析\n预测值: {calc_result}\n"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_003",
                "type": "function",
                "function": {
                    "name": "file_write",
                    "arguments": json.dumps({"path": "report.txt", "content": report_content}, ensure_ascii=False),
                },
            }],
        })
        write_result = _mock_tool_result("file_write", {"path": "report.txt"})
        messages.append({"role": "tool", "tool_call_id": "call_003", "content": write_result})
        messages.append({
            "role": "assistant",
            "content": f"已完成{topic}的研究报告：\n1. 搜索了最新相关信息\n2. 对关键数据进行了计算分析\n3. 将完整报告保存到 report.txt",
        })

    return messages


def _generate_bad_trajectory(template: Dict, filled_query: str) -> List[Dict[str, str]]:
    """
    生成低质量 Agent 轨迹（rejected）。

    坏轨迹的常见问题：
    - 使用了错误的工具
    - 工具参数不正确或不完整
    - 忽略工具返回结果，胡编乱造
    - 回答不完整或跑题
    - 不必要的重复调用
    """
    messages = [
        {"role": "system", "content": template["system_prompt"]},
        {"role": "user", "content": filled_query},
    ]

    task_id = template["id"]
    # 统一的坏模式：选错工具、参数错误、回答敷衍
    bad_patterns = random.choice(["wrong_tool", "bad_params", "no_tool", "hallucinate"])

    if bad_patterns == "wrong_tool":
        # 错误：该搜索时却用计算器
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_bad_001",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"expression": "1+1"}),
                },
            }],
        })
        messages.append({"role": "tool", "tool_call_id": "call_bad_001", "content": "2"})
        messages.append({
            "role": "assistant",
            "content": "结果是2。我已经帮你处理了。",
        })
    elif bad_patterns == "bad_params":
        # 错误：参数为空或不相关
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_bad_001",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"query": ""}),
                },
            }],
        })
        messages.append({"role": "tool", "tool_call_id": "call_bad_001", "content": "搜索查询不能为空"})
        messages.append({
            "role": "assistant",
            "content": "搜索完成了，这里是结果。",  # 错误：无视错误信息
        })
    elif bad_patterns == "no_tool":
        # 错误：该用工具时直接编造答案
        messages.append({
            "role": "assistant",
            "content": "我来直接回答你的问题。根据我的知识，这个问题的答案是...... "
                       "（这里是一段看似合理但可能过时或错误的回答，因为没有使用工具获取最新信息）",
        })
    elif bad_patterns == "hallucinate":
        # 错误：调用了工具但忽略返回结果，编造回答
        tool_name = random.choice(["web_search", "calculator", "get_weather"])
        if tool_name == "web_search":
            args = {"query": "随便搜搜"}
        elif tool_name == "calculator":
            args = {"expression": "1+1"}
        else:
            args = {"city": "北京"}
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_bad_001",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }],
        })
        result = _mock_tool_result(tool_name, args)
        messages.append({"role": "tool", "tool_call_id": "call_bad_001", "content": result})
        messages.append({
            "role": "assistant",
            "content": "根据最新数据显示，全球GDP增长了500%，AI已经完全取代了所有人类工作。"
                       "（注意：这是完全编造的信息，与工具返回的结果无关）",
        })

    return messages


# ============================================================================
# 公开 API
# ============================================================================

def generate_sft_data(num_samples: int = 50, seed: int = 42) -> List[Dict]:
    """
    生成 SFT（有监督微调）训练数据。

    格式：OpenAI 风格的多轮对话，包含 tool_calls 和 tool 角色消息。
    这种格式教会模型 "何时调用工具" 和 "如何正确使用工具"。

    Args:
        num_samples: 生成的样本数
        seed: 随机种子

    Returns:
        list of {"messages": [...], "tools": [...]} 格式的数据
    """
    random.seed(seed)
    dataset = []

    for i in range(num_samples):
        template = random.choice(TASK_TEMPLATES)
        task_id = template["id"]

        # 根据模板填充具体参数
        if task_id == "search_and_summarize":
            topic = random.choice(template["topics"])
            query = template["user_query"].format(topic=topic)
        elif task_id == "math_problem":
            expr = random.choice(template["expressions"])
            query = template["user_query"].format(expression=expr)
        elif task_id == "file_operation":
            fname = random.choice(template["filenames"])
            query = template["user_query"].format(filename=fname)
        elif task_id == "weather_plan":
            city = random.choice(template["cities"])
            query = template["user_query"].format(city=city)
        elif task_id == "multi_step_research":
            topic = random.choice(template["topics"])
            query = template["user_query"].format(topic=topic)
        else:
            continue

        messages = _generate_good_trajectory(template, query)
        dataset.append({
            "id": f"sft_{i:04d}",
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
        })

    return dataset


def generate_dpo_data(num_samples: int = 30, seed: int = 42) -> List[Dict]:
    """
    生成 DPO（直接偏好优化）训练数据。

    格式：每条数据包含 prompt + chosen（好轨迹） + rejected（坏轨迹）。
    DPO 通过对比好/坏轨迹来学习偏好，无需单独的奖励模型。

    核心思想：
    - chosen：正确使用工具、合理推理、完整回答
    - rejected：工具选择错误、参数不当、编造信息

    Args:
        num_samples: 生成的偏好对数
        seed: 随机种子

    Returns:
        list of {"prompt": [...], "chosen": [...], "rejected": [...]} 格式的数据
    """
    random.seed(seed)
    dataset = []

    for i in range(num_samples):
        template = random.choice(TASK_TEMPLATES)
        task_id = template["id"]

        if task_id == "search_and_summarize":
            topic = random.choice(template["topics"])
            query = template["user_query"].format(topic=topic)
        elif task_id == "math_problem":
            expr = random.choice(template["expressions"])
            query = template["user_query"].format(expression=expr)
        elif task_id == "file_operation":
            fname = random.choice(template["filenames"])
            query = template["user_query"].format(filename=fname)
        elif task_id == "weather_plan":
            city = random.choice(template["cities"])
            query = template["user_query"].format(city=city)
        elif task_id == "multi_step_research":
            topic = random.choice(template["topics"])
            query = template["user_query"].format(topic=topic)
        else:
            continue

        good_traj = _generate_good_trajectory(template, query)
        bad_traj = _generate_bad_trajectory(template, query)

        # prompt = system + user，chosen/rejected = 后续的 assistant+tool 轮次
        prompt_messages = [m for m in good_traj if m["role"] in ("system", "user")]
        chosen_messages = [m for m in good_traj if m["role"] not in ("system", "user")]
        rejected_messages = [m for m in bad_traj if m["role"] not in ("system", "user")]

        dataset.append({
            "id": f"dpo_{i:04d}",
            "prompt": prompt_messages,
            "chosen": chosen_messages,
            "rejected": rejected_messages,
            "tools": TOOL_DEFINITIONS,
        })

    return dataset


def save_dataset(data: List[Dict], output_path: str) -> None:
    """将数据集保存为 JSON Lines 格式"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已保存 {len(data)} 条数据到 {output_path}")


# ============================================================================
# 主函数 —— 生成并保存所有训练数据
# ============================================================================

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    # 生成 SFT 数据
    print("=" * 60)
    print("生成 SFT 训练数据（Function Calling 格式）...")
    print("=" * 60)
    sft_data = generate_sft_data(num_samples=50)
    sft_path = os.path.join(output_dir, "sft_agent_data.jsonl")
    save_dataset(sft_data, sft_path)

    # 展示一条样例
    print("\n--- SFT 样例 ---")
    sample = sft_data[0]
    print(f"ID: {sample['id']}")
    print(f"消息轮数: {len(sample['messages'])}")
    for msg in sample["messages"]:
        role = msg["role"]
        if role == "assistant" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]["function"]
            print(f"  [{role}] → 调用工具: {tc['name']}({tc['arguments']})")
        elif role == "tool":
            print(f"  [{role}] → {msg['content'][:80]}...")
        else:
            content = msg.get("content", "")
            if content:
                print(f"  [{role}] {content[:80]}...")

    # 生成 DPO 数据
    print("\n" + "=" * 60)
    print("生成 DPO 偏好数据（Chosen vs Rejected 轨迹）...")
    print("=" * 60)
    dpo_data = generate_dpo_data(num_samples=30)
    dpo_path = os.path.join(output_dir, "dpo_agent_data.jsonl")
    save_dataset(dpo_data, dpo_path)

    # 展示一条样例
    print("\n--- DPO 样例 ---")
    sample = dpo_data[0]
    print(f"ID: {sample['id']}")
    print(f"Prompt 消息数: {len(sample['prompt'])}")
    print(f"Chosen 轮数: {len(sample['chosen'])}")
    print(f"Rejected 轮数: {len(sample['rejected'])}")

    print("\n数据生成完成！")
