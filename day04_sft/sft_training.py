"""
Day 4: SFT 监督微调（Supervised Fine-Tuning）
- 用 trl 的 SFTTrainer 对 Qwen2-0.5B 做 SFT
- 数据集: silk-road/alpaca-data-gpt4-chinese（中文 Alpaca 数据）
- 使用 LoRA（通过 peft 库）减少可训练参数
- Chat Template 处理
- 训练后对比 base model vs SFT model 的回复
"""

import sys
import os
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.experiment_tracker import ExperimentTracker

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Qwen/Qwen2-0.5B"

print(f"Using device: {DEVICE}")

# 初始化实验跟踪器
tracker = ExperimentTracker("day04_sft")

# ============================================================
# 1. 加载模型和 Tokenizer
# ============================================================
print("\n" + "=" * 60)
print("1. 加载 Qwen2-0.5B 模型")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数量: {total_params:,}")

# ============================================================
# 2. LoRA 配置
# ============================================================
"""
LoRA (Low-Rank Adaptation) 原理说明：
- 冻结预训练模型的所有参数
- 在 attention 的 Q、K、V 投影矩阵旁边添加低秩分解矩阵
- 原始权重 W (d×d) 不变，添加 ΔW = BA，其中 B (d×r), A (r×d)
- r 远小于 d，所以可训练参数量大大减少
- 训练时只更新 A 和 B，推理时可以将 ΔW 合并回 W，无额外延迟

关键超参数：
- r (rank): 低秩矩阵的秩，越大表达能力越强但参数越多
- lora_alpha: 缩放因子，通常设为 r 的 2 倍
- target_modules: 要添加 LoRA 的模块（通常是 Q、K、V 投影）
"""
print("\n" + "=" * 60)
print("2. 配置 LoRA")
print("=" * 60)

lora_config = LoraConfig(
    r=16,                           # 低秩矩阵的秩
    lora_alpha=32,                  # 缩放因子 (alpha/r 决定 LoRA 的影响程度)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 对 attention 矩阵做 LoRA
    lora_dropout=0.05,              # LoRA 层的 dropout
    bias="none",                    # 不训练 bias
    task_type=TaskType.CAUSAL_LM,   # 因果语言模型任务
)

model = get_peft_model(model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
model.print_trainable_parameters()

import json
tracker.log_text("config", json.dumps({
    "model_name": MODEL_NAME, "total_params": all_params,
    "trainable_params": trainable_params, "lora_r": 16,
    "lora_alpha": 32, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}, ensure_ascii=False, indent=2))

# ============================================================
# 3. 加载和处理数据集
# ============================================================
print("\n" + "=" * 60)
print("3. 加载中文 Alpaca 数据集")
print("=" * 60)

dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", trust_remote_code=True)
print(f"数据集大小: {len(dataset['train'])}")
print(f"样本示例:\n{dataset['train'][0]}")

# 只取一部分数据做演示（完整训练需要更多时间）
train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
print(f"使用训练数据: {len(train_dataset)} 条")


def format_alpaca_to_chat(example):
    """
    将 Alpaca 格式数据转换为 chat 格式

    Chat Template 处理说明：
    - 现代 LLM 都使用特定的 chat template 来区分不同角色的对话
    - Qwen2 使用 <|im_start|>role\ncontent<|im_end|> 格式
    - 正确使用 chat template 对 SFT 效果至关重要
    - tokenizer.apply_chat_template() 会自动应用模型对应的模板
    """
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    # 构建用户消息
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    # 使用 tokenizer 的 chat template 格式化
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": formatted}


print("格式化数据集...")
train_dataset = train_dataset.map(format_alpaca_to_chat, remove_columns=train_dataset.column_names)
print(f"格式化后的样本:\n{train_dataset[0]['text'][:500]}")

# ============================================================
# 4. 配置 SFTTrainer 并训练
# ============================================================
"""
Loss Masking 原理说明：
- 在 SFT 中，我们只想让模型学习"回复"部分，不需要学习"指令"部分
- Loss masking 的作用：只在 assistant 回复的 token 上计算 loss
- 具体实现：将 instruction/user 部分的 label 设为 -100（PyTorch 的 ignore_index）
- 这样模型不会因为"学不好指令部分"而受到惩罚
- trl 的 SFTTrainer 在使用 chat template 时会自动处理 loss masking

为什么需要 Loss Masking？
1. 指令部分是固定的输入，模型不需要"生成"它
2. 如果对指令部分也计算 loss，会稀释模型学习回复的能力
3. 相当于告诉模型：你只需要学会"怎么回答"，不需要学会"怎么提问"
"""
print("\n" + "=" * 60)
print("4. 开始 SFT 训练")
print("=" * 60)

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_output")

sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,       # 等效 batch_size = 4 * 4 = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=20,
    save_strategy="epoch",
    max_seq_length=512,
    fp16=torch.cuda.is_available(),
    report_to="none",
    dataset_text_field="text",           # 指定文本字段名
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

# --- 训练前：先用 base model 生成回复 ---
print("\n训练前 base model 的回复:")
test_prompts = [
    "请解释什么是机器学习？",
    "写一首关于春天的诗。",
    "如何学好编程？",
]

pre_sft_responses = []
model.eval()
for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=150, temperature=0.7,
            do_sample=True, top_k=50, pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pre_sft_responses.append(response)
    print(f"\n  问: {prompt}")
    print(f"  答: {response[:200]}")

# --- 开始训练 ---
start_time = time.time()
train_result = trainer.train()
elapsed = time.time() - start_time
print(f"\n训练完成! 耗时: {elapsed:.0f}s")

tracker.log_metric("train_loss", train_result.metrics["train_loss"])
tracker.log_metric("training_time_seconds", elapsed)

# --- 训练后：用 SFT model 生成回复 ---
print("\n训练后 SFT model 的回复:")
post_sft_responses = []
model.eval()
for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=150, temperature=0.7,
            do_sample=True, top_k=50, pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    post_sft_responses.append(response)
    print(f"\n  问: {prompt}")
    print(f"  答: {response[:200]}")

# ============================================================
# 5. 保存对比结果到实验记录
# ============================================================
print("\n" + "=" * 60)
print("5. 保存对比结果")
print("=" * 60)

for i, prompt in enumerate(test_prompts):
    tracker.log_text(f"comparison_{i}", (
        f"问题: {prompt}\n\n"
        f"--- SFT 前 ---\n{pre_sft_responses[i]}\n\n"
        f"--- SFT 后 ---\n{post_sft_responses[i]}"
    ))

print("\n实验摘要:")
print(json.dumps(tracker.summary(), indent=2, ensure_ascii=False))
print("\nDay 4 SFT 训练完成！")
