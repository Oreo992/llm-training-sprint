"""
Day 5: 数据质量实验 - 少量高质量 vs 大量低质量
- 构造高质量和低质量 SFT 数据的对比实验
- 提供数据生成模板（可用 API 生成高质量数据）
- 对比训练：少量高质量 vs 大量低质量
- 自动评估和对比，保存到实验记录
"""

import sys
import os
import json
import random
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.experiment_tracker import ExperimentTracker

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Qwen/Qwen2-0.5B"

print(f"Using device: {DEVICE}")

# 初始化实验跟踪器
tracker = ExperimentTracker("day05_data_quality")

# ============================================================
# 1. 构造高质量和低质量数据
# ============================================================
print("\n" + "=" * 60)
print("1. 构造对比数据集")
print("=" * 60)

"""
数据质量维度说明：
- 高质量数据：回答准确、结构清晰、内容丰富、有逻辑性
- 低质量数据：回答笼统、有错误、缺乏细节、逻辑混乱

实际项目中，高质量数据可以通过以下方式获得：
1. 人工标注（成本最高但质量最好）
2. 用强模型（如 GPT-4）生成，再人工审核
3. 从已有数据中筛选高质量样本
"""

# --- 高质量数据模板 ---
HIGH_QUALITY_DATA = [
    {
        "instruction": "解释什么是梯度下降算法？",
        "output": "梯度下降是一种迭代优化算法，用于找到函数的最小值。其核心思想是：\n\n1. **基本原理**：沿着函数梯度（导数）的反方向移动，因为梯度指向函数增长最快的方向，反方向就是下降最快的方向。\n\n2. **更新公式**：θ = θ - α × ∇J(θ)，其中 α 是学习率，∇J(θ) 是损失函数的梯度。\n\n3. **常见变体**：\n   - 批量梯度下降（BGD）：使用全部数据计算梯度，稳定但慢\n   - 随机梯度下降（SGD）：每次用一个样本，快但不稳定\n   - 小批量梯度下降（Mini-batch SGD）：折中方案，最常用\n\n4. **学习率的重要性**：太大会导致震荡甚至发散，太小会收敛很慢。现代方法如 Adam 会自适应调整学习率。"
    },
    {
        "instruction": "Python 中列表和元组有什么区别？",
        "output": "列表（list）和元组（tuple）是 Python 中两种常用的序列类型，主要区别如下：\n\n1. **可变性**：列表是可变的（mutable），可以增删改元素；元组是不可变的（immutable），创建后不能修改。\n\n2. **语法**：列表用方括号 `[1, 2, 3]`，元组用圆括号 `(1, 2, 3)`。\n\n3. **性能**：元组因为不可变，在创建和访问时比列表更快，占用内存更少。\n\n4. **使用场景**：\n   - 列表：需要频繁修改的数据集合，如购物车商品\n   - 元组：表示固定的数据组合，如坐标 (x, y)、函数返回多个值\n\n5. **作为字典键**：元组可以作为字典的键（因为不可变），列表不行。\n\n```python\nmy_list = [1, 2, 3]    # 可以 append、remove\nmy_tuple = (1, 2, 3)   # 创建后不能修改\n```"
    },
    {
        "instruction": "什么是过拟合？如何防止？",
        "output": "过拟合（Overfitting）是指模型在训练数据上表现很好，但在新数据上表现差的现象。本质是模型"记住"了训练数据的噪声，而非学到真正的规律。\n\n**判断方法**：训练损失持续下降，但验证损失开始上升。\n\n**常见防止方法**：\n\n1. **增加数据量**：最直接有效的方法，更多数据让模型更难记住噪声\n2. **数据增强**：对现有数据做变换（翻转、裁剪、加噪声等）\n3. **正则化**：\n   - L1/L2 正则化：限制权重大小\n   - Dropout：训练时随机丢弃部分神经元\n4. **早停（Early Stopping）**：当验证集性能不再提升时停止训练\n5. **降低模型复杂度**：减少参数量，使用更简单的模型\n6. **Batch Normalization**：稳定训练过程，有一定正则化效果\n7. **交叉验证**：更可靠地评估模型泛化能力"
    },
    {
        "instruction": "简述 HTTP 和 HTTPS 的区别。",
        "output": "HTTP 和 HTTPS 的核心区别在于安全性：\n\n1. **加密**：HTTP 以明文传输数据，任何人都可以截获内容；HTTPS 使用 TLS/SSL 协议加密，数据在传输过程中是加密的。\n\n2. **端口**：HTTP 默认使用 80 端口，HTTPS 默认使用 443 端口。\n\n3. **证书**：HTTPS 需要 SSL 证书（由 CA 机构颁发），用于验证服务器身份，防止中间人攻击。\n\n4. **性能**：HTTPS 因为加解密操作，会有少量性能开销，但现代硬件下几乎可以忽略。\n\n5. **SEO**：搜索引擎（如 Google）优先收录 HTTPS 网站。\n\n**实际建议**：所有网站都应该使用 HTTPS，尤其是涉及用户登录、支付等敏感操作的场景。Let's Encrypt 提供免费的 SSL 证书。"
    },
    {
        "instruction": "什么是 Docker？它有什么用？",
        "output": "Docker 是一种容器化技术，可以将应用及其所有依赖打包成一个标准化的单元（容器），确保在任何环境中都能一致运行。\n\n**核心概念**：\n1. **镜像（Image）**：只读模板，包含应用代码和运行环境\n2. **容器（Container）**：镜像的运行实例，相互隔离\n3. **Dockerfile**：定义如何构建镜像的脚本\n\n**主要用途**：\n1. **环境一致性**：解决"在我机器上能跑"的问题\n2. **快速部署**：秒级启动，比虚拟机轻量得多\n3. **微服务架构**：每个服务独立容器化\n4. **CI/CD**：标准化构建和测试环境\n\n**与虚拟机的区别**：\n- 虚拟机：虚拟化硬件，运行完整 OS，启动慢，资源占用大\n- Docker：共享宿主 OS 内核，只隔离应用层，启动快，资源占用小"
    },
]

# --- 低质量数据模板（常见问题）---
LOW_QUALITY_DATA = [
    {"instruction": "解释什么是梯度下降算法？", "output": "梯度下降就是一种算法。"},
    {"instruction": "Python 中列表和元组有什么区别？", "output": "列表和元组差不多，一个用方括号一个用圆括号。"},
    {"instruction": "什么是过拟合？如何防止？", "output": "过拟合就是训练过头了，用正则化可以解决。"},
    {"instruction": "简述 HTTP 和 HTTPS 的区别。", "output": "HTTPS 比 HTTP 多了个 S，更安全。"},
    {"instruction": "什么是 Docker？它有什么用？", "output": "Docker 是一个工具，用来运行程序的。"},
    {"instruction": "什么是数据库索引？", "output": "索引能让查询变快。"},
    {"instruction": "什么是 RESTful API？", "output": "就是一种 API 风格。"},
    {"instruction": "什么是递归？", "output": "递归就是函数调用自己。"},
    {"instruction": "什么是多线程？", "output": "多线程就是同时运行多个线程。"},
    {"instruction": "什么是 Git？", "output": "Git 是版本管理工具。"},
    {"instruction": "什么是面向对象编程？", "output": "面向对象就是用类和对象来编程。"},
    {"instruction": "Python 的装饰器是什么？", "output": "装饰器就是在函数上面加 @ 符号。"},
    {"instruction": "什么是 SQL 注入？", "output": "SQL 注入是一种攻击。"},
    {"instruction": "什么是 TCP 和 UDP 的区别？", "output": "TCP 可靠，UDP 不可靠。"},
    {"instruction": "什么是缓存？", "output": "缓存就是把数据放在内存里。"},
]

# 额外的高质量数据（补充到合理数量）
EXTRA_HIGH_QUALITY = [
    {
        "instruction": "什么是数据库索引？",
        "output": "数据库索引是一种数据结构，用于加速数据库查询操作。\n\n**工作原理**：类似书籍的目录，通过索引可以快速定位到数据位置，而不需要扫描整张表。\n\n**常见类型**：\n1. **B-Tree 索引**：最常用，适合范围查询和等值查询\n2. **Hash 索引**：只适合等值查询，速度极快\n3. **全文索引**：用于文本搜索\n4. **复合索引**：多列组合索引，注意最左前缀原则\n\n**使用建议**：\n- 在 WHERE、JOIN、ORDER BY 常用的列上建索引\n- 不要过度索引，索引会占用存储空间且影响写入性能\n- 区分度低的列（如性别）不适合单独建索引\n\n**代价**：索引会增加写操作的开销（INSERT/UPDATE/DELETE 需要同时更新索引），并占用额外的磁盘空间。"
    },
    {
        "instruction": "什么是 RESTful API？",
        "output": "RESTful API 是一种基于 REST（Representational State Transfer）架构风格设计的 Web API。\n\n**核心原则**：\n1. **资源导向**：URL 表示资源，如 `/users/123` 表示 ID 为 123 的用户\n2. **HTTP 方法语义化**：\n   - GET：获取资源\n   - POST：创建资源\n   - PUT：更新资源（完整替换）\n   - PATCH：部分更新\n   - DELETE：删除资源\n3. **无状态**：每个请求包含所有必要信息，服务器不保存会话状态\n4. **统一接口**：资源的操作方式一致\n\n**最佳实践**：\n- URL 用名词复数：`/api/users` 而非 `/api/getUsers`\n- 用 HTTP 状态码表示结果：200（成功）、404（不存在）、500（服务器错误）\n- 支持分页、筛选、排序\n- 版本控制：`/api/v1/users`"
    },
    {
        "instruction": "什么是递归？",
        "output": "递归是一种编程技巧，指函数在其定义中调用自身来解决问题。\n\n**核心要素**：\n1. **基线条件（Base Case）**：递归终止的条件，防止无限递归\n2. **递归步骤**：将问题分解为更小的子问题\n\n**经典例子 - 阶乘**：\n```python\ndef factorial(n):\n    if n <= 1:        # 基线条件\n        return 1\n    return n * factorial(n - 1)  # 递归步骤\n```\n\n**优缺点**：\n- 优点：代码简洁，适合树形结构、分治问题\n- 缺点：栈溢出风险，重复计算（可用记忆化解决）\n\n**常见应用**：斐波那契数列、二叉树遍历、快速排序、汉诺塔\n\n**注意**：Python 默认递归深度限制为 1000 层，可通过 `sys.setrecursionlimit()` 修改，但更好的做法是改写为迭代。"
    },
    {
        "instruction": "什么是多线程？",
        "output": "多线程是一种并发编程模型，允许程序同时执行多个线程（执行路径）。\n\n**基本概念**：\n- **进程**：操作系统分配资源的基本单位，有独立内存空间\n- **线程**：进程内的执行单元，共享进程的内存空间\n\n**优势**：\n1. 提高 I/O 密集型任务的效率（如网络请求、文件读写）\n2. 保持程序响应性（如 UI 线程 + 后台工作线程）\n\n**挑战**：\n1. **竞态条件**：多个线程同时修改共享数据导致结果不确定\n2. **死锁**：线程互相等待对方释放资源\n3. **GIL（Python 特有）**：全局解释器锁使得 Python 线程无法真正并行执行 CPU 密集型任务\n\n**Python 中的选择**：\n- I/O 密集型：使用 `threading` 或 `asyncio`\n- CPU 密集型：使用 `multiprocessing`（多进程绕过 GIL）"
    },
    {
        "instruction": "什么是 Git？",
        "output": "Git 是一个分布式版本控制系统，用于跟踪文件变更、协调多人协作开发。\n\n**核心概念**：\n1. **仓库（Repository）**：项目的所有文件和历史记录\n2. **提交（Commit）**：一次代码快照，包含作者、时间、变更内容\n3. **分支（Branch）**：独立的开发线路，互不影响\n4. **合并（Merge）**：将分支的改动合并到主线\n\n**常用工作流**：\n```bash\ngit clone <url>        # 克隆远程仓库\ngit checkout -b feature  # 创建并切换分支\ngit add .              # 暂存所有修改\ngit commit -m \"描述\"    # 提交\ngit push origin feature  # 推送到远程\n```\n\n**为什么分布式？** 每个开发者都有完整的仓库副本，即使没有网络也能提交、查看历史。\n\n**常用平台**：GitHub、GitLab、Gitee"
    },
]

# 合并高质量数据
all_high_quality = HIGH_QUALITY_DATA + EXTRA_HIGH_QUALITY
print(f"高质量数据: {len(all_high_quality)} 条")
print(f"低质量数据: {len(LOW_QUALITY_DATA)} 条")

tracker.log_text("config", json.dumps({
    "model_name": MODEL_NAME, "high_quality_count": len(all_high_quality),
    "low_quality_count": len(LOW_QUALITY_DATA), "experiment": "少量高质量 vs 大量低质量",
}, ensure_ascii=False, indent=2))


# ============================================================
# 2. 数据生成模板（用 API 生成高质量数据）
# ============================================================
"""
数据生成模板说明：
在实际项目中，可以通过调用强模型 API 来生成高质量训练数据。

示例 prompt 模板：

```python
GENERATION_PROMPT = '''
你是一位资深技术专家。请针对以下问题给出高质量的回答。
要求：
1. 回答准确、有深度
2. 使用清晰的结构（标题、列表、代码示例等）
3. 包含实际案例或最佳实践
4. 适合初学者理解
5. 300-500 字

问题：{instruction}
'''

# 使用 OpenAI API 生成
import openai
client = openai.OpenAI(api_key="your-key")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": GENERATION_PROMPT.format(instruction=q)}],
    temperature=0.7,
)
high_quality_answer = response.choices[0].message.content
```

数据质量检查清单：
- [ ] 事实准确性：答案中的信息是否正确？
- [ ] 完整性：是否覆盖了问题的主要方面？
- [ ] 结构性：是否有清晰的组织结构？
- [ ] 可读性：是否易于理解？
- [ ] 一致性：风格和格式是否统一？
"""


# ============================================================
# 3. 准备训练数据
# ============================================================
print("\n" + "=" * 60)
print("2. 准备训练数据")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_to_chat(examples):
    """将指令数据转换为 chat 格式"""
    texts = []
    for inst, out in zip(examples["instruction"], examples["output"]):
        messages = [
            {"role": "user", "content": inst},
            {"role": "assistant", "content": out},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts}


# 构造 Dataset
hq_dataset = Dataset.from_list(all_high_quality)
lq_dataset = Dataset.from_list(LOW_QUALITY_DATA)

hq_dataset = hq_dataset.map(format_to_chat, batched=True, remove_columns=hq_dataset.column_names)
lq_dataset = lq_dataset.map(format_to_chat, batched=True, remove_columns=lq_dataset.column_names)

print(f"高质量数据集: {len(hq_dataset)} 条")
print(f"低质量数据集: {len(lq_dataset)} 条")


# ============================================================
# 4. 对比训练函数
# ============================================================

def train_model(train_dataset, experiment_name, num_epochs=3):
    """
    训练一个模型并返回训练后的模型

    参数:
    - train_dataset: 训练数据集
    - experiment_name: 实验名称
    - num_epochs: 训练轮数
    """
    print(f"\n--- 训练: {experiment_name} ---")

    # 每次训练都重新加载模型（确保公平对比）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # LoRA 配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"output_{experiment_name}"
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="no",
        max_seq_length=512,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    start_time = time.time()
    result = trainer.train()
    elapsed = time.time() - start_time

    print(f"  训练完成! Loss: {result.metrics['train_loss']:.4f}, 耗时: {elapsed:.0f}s")
    return model, result.metrics["train_loss"], elapsed


# ============================================================
# 5. 执行对比实验
# ============================================================
print("\n" + "=" * 60)
print("3. 执行对比实验")
print("=" * 60)

# 训练两个模型
hq_model, hq_loss, hq_time = train_model(hq_dataset, "high_quality", num_epochs=5)
lq_model, lq_loss, lq_time = train_model(lq_dataset, "low_quality", num_epochs=5)

tracker.log_metric("hq_train_loss", hq_loss)
tracker.log_metric("lq_train_loss", lq_loss)
tracker.log_metric("hq_training_time", hq_time)
tracker.log_metric("lq_training_time", lq_time)

# ============================================================
# 6. 自动评估和对比
# ============================================================
print("\n" + "=" * 60)
print("4. 自动评估和对比")
print("=" * 60)

# 评估问题（既包含训练中见过的主题，也有新主题）
eval_prompts = [
    "什么是梯度下降？请简要说明。",          # 训练中见过的主题
    "解释 Python 中 list 和 tuple 的区别。",  # 训练中见过的主题
    "什么是微服务架构？",                     # 新主题
    "请解释什么是设计模式中的单例模式？",      # 新主题
]


def generate_response(model, prompt, max_new_tokens=200):
    """用模型生成回复"""
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def simple_quality_score(response):
    """
    简单的质量评分（0-100）

    评估维度：
    1. 长度：太短的回答通常质量低
    2. 结构：是否有换行、列表等结构化内容
    3. 细节：是否包含具体的技术细节标志词
    """
    score = 0

    # 长度评分（0-30分）
    length = len(response)
    if length > 300:
        score += 30
    elif length > 150:
        score += 20
    elif length > 50:
        score += 10

    # 结构评分（0-30分）
    if "\n" in response:
        score += 10
    if any(marker in response for marker in ["1.", "2.", "- ", "* ", "**"]):
        score += 10
    if "```" in response:
        score += 10

    # 内容深度评分（0-40分）
    detail_keywords = [
        "例如", "比如", "原因", "因为", "优点", "缺点", "区别",
        "原理", "实现", "步骤", "注意", "建议", "场景", "应用",
        "优势", "劣势", "对比", "总结",
    ]
    keyword_count = sum(1 for kw in detail_keywords if kw in response)
    score += min(keyword_count * 8, 40)

    return min(score, 100)


print("\n对比结果:")
print("-" * 80)

hq_total_score = 0
lq_total_score = 0

for prompt in eval_prompts:
    hq_response = generate_response(hq_model, prompt)
    lq_response = generate_response(lq_model, prompt)

    hq_score = simple_quality_score(hq_response)
    lq_score = simple_quality_score(lq_response)

    hq_total_score += hq_score
    lq_total_score += lq_score

    print(f"\n问题: {prompt}")
    print(f"\n  [高质量模型] (得分: {hq_score})")
    print(f"  {hq_response[:300]}")
    print(f"\n  [低质量模型] (得分: {lq_score})")
    print(f"  {lq_response[:300]}")
    print("-" * 80)

    # 保存到实验记录
    tracker.log_text(f"eval_{prompt[:10]}", (
        f"问题: {prompt}\n\n"
        f"[高质量模型] (得分: {hq_score})\n{hq_response[:500]}\n\n"
        f"[低质量模型] (得分: {lq_score})\n{lq_response[:500]}"
    ))

# 汇总
hq_avg = hq_total_score / len(eval_prompts)
lq_avg = lq_total_score / len(eval_prompts)

tracker.log_metric("hq_avg_quality_score", hq_avg)
tracker.log_metric("lq_avg_quality_score", lq_avg)

print(f"\n{'=' * 60}")
print(f"实验结论:")
print(f"  高质量数据模型 - 平均质量得分: {hq_avg:.1f} (数据量: {len(all_high_quality)} 条)")
print(f"  低质量数据模型 - 平均质量得分: {lq_avg:.1f} (数据量: {len(LOW_QUALITY_DATA)} 条)")
print(f"  得分差异: {hq_avg - lq_avg:+.1f}")
if hq_avg > lq_avg:
    print(f"  结论: 少量高质量数据优于大量低质量数据！")
else:
    print(f"  结论: 本次实验中低质量数据模型得分更高，可能需要更多轮训练或调整评估标准。")
print(f"{'=' * 60}")

# ============================================================
# 7. 保存实验结果
# ============================================================
print("\n实验摘要:")
print(json.dumps(tracker.summary(), indent=2, ensure_ascii=False))
print("\nDay 5 数据质量实验完成！")
