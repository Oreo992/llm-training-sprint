"""
Day 1: 用 PyTorch 从零写一个文本分类模型（不用 HuggingFace Trainer）
- 数据集：IMDb 情感分类
- 模型：2 层 Transformer Encoder + 分类头
- 手写 training loop: forward → loss → backward → step
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import os
import time

# ============================================================
# 1. 配置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
MAX_LEN = 256
VOCAB_SIZE = 30522  # bert tokenizer vocab size
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
NUM_CLASSES = 2

print(f"Using device: {DEVICE}")

# ============================================================
# 2. 数据加载与预处理
# ============================================================
print("Loading IMDb dataset...")
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


class IMDbDataset(Dataset):
    """将 IMDb 数据转成 PyTorch Dataset"""

    def __init__(self, hf_dataset, tokenizer, max_len):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


train_dataset = IMDbDataset(dataset["train"], tokenizer, MAX_LEN)
test_dataset = IMDbDataset(dataset["test"], tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# ============================================================
# 3. 模型定义：2 层 Transformer Encoder + 分类头
# ============================================================


class TransformerClassifier(nn.Module):
    """
    核心架构：
    1. Token Embedding + Positional Embedding
    2. 2 层 TransformerEncoder（self-attention + FFN）
    3. 全局平均池化
    4. 分类头（Linear → 2类）
    """

    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # attention_mask: 1=valid, 0=pad → TransformerEncoder 需要 src_key_padding_mask: True=pad
        padding_mask = (attention_mask == 0)

        # Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # 全局平均池化（只在非 pad 位置上做）
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        # 分类
        logits = self.classifier(x)
        return logits


model = TransformerClassifier(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_CLASSES, MAX_LEN)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ============================================================
# 4. 训练组件
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ============================================================
# 5. 手写 Training Loop
# ============================================================
train_losses = []
val_accuracies = []


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """一个完整的训练 epoch：forward → loss → backward → step"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        logits = model(input_ids, attention_mask)

        # 计算 loss
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 参数更新
        optimizer.step()

        # 统计
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = correct / total * 100
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(loader)} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")

    avg_loss = total_loss / len(loader)
    acc = correct / total * 100
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device):
    """验证集评估"""
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total * 100


# ============================================================
# 6. 训练主循环
# ============================================================
print("\n" + "=" * 60)
print("开始训练")
print("=" * 60)

for epoch in range(EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
    val_acc = evaluate(model, test_loader, DEVICE)

    elapsed = time.time() - start_time
    train_losses.append(train_loss)
    val_accuracies.append(val_acc)

    print(f"\nEpoch {epoch+1}/{EPOCHS} | Time: {elapsed:.0f}s")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
    print(f"  Val Acc: {val_acc:.1f}%")
    print()

# ============================================================
# 7. 保存结果
# ============================================================
# 保存模型
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/imdb_transformer.pt")

# 画 loss 曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, EPOCHS + 1), train_losses, marker="o")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")

ax2.plot(range(1, EPOCHS + 1), val_accuracies, marker="o", color="green")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Validation Accuracy")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print(f"\n训练完成！最终验证准确率: {val_accuracies[-1]:.1f}%")
print("Loss 曲线已保存到 training_curves.png")
