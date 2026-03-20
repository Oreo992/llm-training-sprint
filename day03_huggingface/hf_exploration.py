"""
Day 3: HuggingFace 生态探索
- 加载 Qwen2-0.5B 模型和 tokenizer，分析其结构
- 分析 tokenizer：vocab size、special tokens、chat template
- 用 datasets 加载数据集并做预处理 pipeline
- 用 HF Trainer 做文本分类 fine-tune
- 保存微调前后的预测结果对比
"""

import sys
import os
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.experiment_tracker import ExperimentTracker

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Qwen/Qwen2-0.5B"

print(f"Using device: {DEVICE}")

# 初始化实验跟踪器
tracker = ExperimentTracker("day03_hf_exploration")

# ============================================================
# 1. 加载 Qwen2-0.5B 模型和 Tokenizer
# ============================================================
print("\n" + "=" * 60)
print("1. 加载 Qwen2-0.5B 模型和 Tokenizer")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model_causal = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16
).to(DEVICE)

total_params = sum(p.numel() for p in model_causal.parameters())
print(f"模型参数量: {total_params:,}")

# ============================================================
# 2. 分析 Tokenizer
# ============================================================
print("\n" + "=" * 60)
print("2. 分析 Tokenizer")
print("=" * 60)

print(f"词表大小 (vocab_size): {tokenizer.vocab_size}")
print(f"模型最大长度: {tokenizer.model_max_length}")
print(f"特殊 tokens:")
print(f"  PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"  EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
print(f"  BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
print(f"  UNK token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")

# 测试 tokenizer 的编解码
test_texts = [
    "Hello, world!",
    "你好，世界！",
    "机器学习是人工智能的一个分支。",
]
print("\n编解码测试:")
for text in test_texts:
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    token_strs = tokenizer.convert_ids_to_tokens(tokens)
    print(f"  原文: {text}")
    print(f"  Token IDs: {tokens}")
    print(f"  Token 字符串: {token_strs}")
    print(f"  解码: {decoded}")
    print()

# Chat Template 分析
print("Chat Template:")
if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
    print(f"  模板内容:\n{tokenizer.chat_template[:500]}")
    # 测试 chat template
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "什么是机器学习？"},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"\n  应用 chat template 后:\n{formatted}")
else:
    print("  该模型没有 chat template")

tokenizer_info = {
    "vocab_size": tokenizer.vocab_size,
    "model_max_length": tokenizer.model_max_length,
    "pad_token": str(tokenizer.pad_token),
    "eos_token": str(tokenizer.eos_token),
    "has_chat_template": hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None,
}
import json
tracker.log_text("config", json.dumps({"model_name": MODEL_NAME, "total_params": total_params, "tokenizer": tokenizer_info}, ensure_ascii=False, indent=2))

# ============================================================
# 3. 用 datasets 加载数据集并做预处理
# ============================================================
print("\n" + "=" * 60)
print("3. 加载数据集并做预处理 pipeline")
print("=" * 60)

# 使用较小的情感分类数据集做演示
dataset = load_dataset("yelp_review_full", trust_remote_code=True)
print(f"数据集结构: {dataset}")
print(f"训练集大小: {len(dataset['train'])}")
print(f"样本: {dataset['train'][0]}")

# 预处理 pipeline：只取一小部分数据做演示
small_train = dataset["train"].shuffle(seed=42).select(range(2000))
small_test = dataset["test"].shuffle(seed=42).select(range(500))

# 设置 pad_token（Qwen2 可能没有 pad_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    """
    预处理 pipeline：
    1. 对文本做 tokenize
    2. 截断到固定长度
    3. 添加 padding
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

print("开始 tokenize 数据...")
tokenized_train = small_train.map(preprocess_function, batched=True, remove_columns=["text"])
tokenized_test = small_test.map(preprocess_function, batched=True, remove_columns=["text"])

# 重命名 label 列
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_test = tokenized_test.rename_column("label", "labels")
tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

print(f"预处理完成! 训练集: {len(tokenized_train)}, 测试集: {len(tokenized_test)}")

# ============================================================
# 4. 用 HF Trainer 做文本分类 fine-tune
# ============================================================
print("\n" + "=" * 60)
print("4. 文本分类 Fine-tune（使用 HF Trainer）")
print("=" * 60)

# 加载分类模型（5 个类别：1-5 星评分）
NUM_LABELS = 5
model_cls = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    trust_remote_code=True,
    torch_dtype=torch.float32,  # 分类任务用 float32 更稳定
)
model_cls.config.pad_token_id = tokenizer.pad_token_id

# --- 微调前：先看看随机初始化的分类头效果 ---
print("\n微调前的预测结果（随机分类头）:")
model_cls.eval()
model_cls.to(DEVICE)

test_texts_cls = [
    "This restaurant is absolutely amazing! Best food ever!",
    "Terrible experience. The food was cold and the service was awful.",
    "It was okay, nothing special but not bad either.",
]

pre_finetune_predictions = []
for text in test_texts_cls:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model_cls(**inputs)
    pred = outputs.logits.argmax(dim=-1).item()
    pre_finetune_predictions.append({"text": text, "predicted_label": pred})
    print(f"  文本: {text[:60]}...")
    print(f"  预测: {pred + 1} 星")

# --- 定义评估指标 ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# --- 训练配置 ---
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_trainer_output")
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",  # 不用 wandb 等外部工具
    fp16=torch.cuda.is_available(),
)

# --- 创建 Trainer ---
trainer = Trainer(
    model=model_cls,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# --- 开始训练 ---
print("\n开始 fine-tune...")
start_time = time.time()
train_result = trainer.train()
elapsed = time.time() - start_time
print(f"训练完成! 耗时: {elapsed:.0f}s")

# 评估
eval_result = trainer.evaluate()
print(f"测试集准确率: {eval_result['eval_accuracy']:.4f}")

tracker.log_metric("train_loss", train_result.metrics["train_loss"])
tracker.log_metric("eval_accuracy", eval_result["eval_accuracy"])
tracker.log_metric("training_time_seconds", elapsed)

# --- 微调后：再看预测结果 ---
print("\n微调后的预测结果:")
model_cls.eval()
post_finetune_predictions = []
for text in test_texts_cls:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model_cls(**inputs)
    pred = outputs.logits.argmax(dim=-1).item()
    post_finetune_predictions.append({"text": text, "predicted_label": pred})
    print(f"  文本: {text[:60]}...")
    print(f"  预测: {pred + 1} 星")

# ============================================================
# 5. 保存对比结果到实验记录
# ============================================================
print("\n" + "=" * 60)
print("5. 保存对比结果")
print("=" * 60)

for i, text in enumerate(test_texts_cls):
    tracker.log_text(f"comparison_{i}", (
        f"文本: {text}\n"
        f"微调前预测: {pre_finetune_predictions[i]['predicted_label'] + 1} 星\n"
        f"微调后预测: {post_finetune_predictions[i]['predicted_label'] + 1} 星"
    ))

print("\n实验摘要:")
print(json.dumps(tracker.summary(), indent=2, ensure_ascii=False))
print("\nDay 3 完成！")
