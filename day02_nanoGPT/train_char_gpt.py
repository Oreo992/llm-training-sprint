"""
Day 2: 字符级 GPT 训练（nanoGPT）
- 从零实现一个小型 GPT：MultiHeadAttention, TransformerBlock, GPT
- 使用 Shakespeare 数据集（tiny_shakespeare）
- Causal mask：每个 token 只能看到自己和前面的 token
- 手写 training loop
- 训练完成后生成文本展示效果
"""

import sys
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 添加项目根目录到 sys.path，以便 import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.experiment_tracker import ExperimentTracker

# ============================================================
# 1. 配置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4
BLOCK_SIZE = 128       # 序列长度（上下文窗口）
D_MODEL = 128          # 嵌入维度
N_HEAD = 4             # 注意力头数
N_LAYER = 4            # Transformer 层数
DROPOUT = 0.1

print(f"Using device: {DEVICE}")

# ============================================================
# 2. 下载 Shakespeare 数据集
# ============================================================
print("加载 tiny_shakespeare 数据集...")
from datasets import load_dataset

ds = load_dataset("tiny_shakespeare", trust_remote_code=True)
# tiny_shakespeare 只有 train/validation/test，把文本拼起来
text = ds["train"]["text"][0]
print(f"数据集字符数: {len(text):,}")
print(f"前 200 个字符:\n{text[:200]}")

# ============================================================
# 3. 字符级 Tokenizer
# ============================================================
# 收集所有出现过的字符作为词表
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
print(f"词表大小（不同字符数）: {VOCAB_SIZE}")

# 字符 <-> 索引 的映射
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    """将字符串编码为整数列表"""
    return [stoi[c] for c in s]

def decode(ids):
    """将整数列表解码为字符串"""
    return "".join([itos[i] for i in ids])

# 编码整个数据集
data = torch.tensor(encode(text), dtype=torch.long)
print(f"编码后的 tensor 形状: {data.shape}")

# 划分训练集和验证集（90% / 10%）
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# ============================================================
# 4. Dataset
# ============================================================
class CharDataset(Dataset):
    """
    字符级数据集：
    每个样本是连续的 BLOCK_SIZE 个字符作为输入（x），
    对应的目标（y）是右移一位的序列（预测下一个字符）
    """
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


train_dataset = CharDataset(train_data, BLOCK_SIZE)
val_dataset = CharDataset(val_data, BLOCK_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================
# 5. 模型定义：从零实现 GPT
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    多头自注意力（Multi-Head Self-Attention）

    核心公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    关键点：
    - 将 Q, K, V 分成多个头，每个头独立计算注意力
    - 使用 causal mask 确保每个位置只能看到自己和之前的位置（自回归特性）
    - 最后将多个头的输出拼接起来
    """
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        self.n_head = n_head
        self.d_head = d_model // n_head  # 每个头的维度

        # Q, K, V 投影矩阵（合并成一个大矩阵提高效率）
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # batch_size, seq_len, d_model

        # 计算 Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # 每个: (B, T, C)

        # 拆分多头: (B, T, C) -> (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # 计算注意力分数: (B, n_head, T, T)
        scale = math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal Mask（因果掩码）：上三角矩阵设为 -inf
        # 这确保位置 i 只能关注位置 0..i，不能"偷看"未来的 token
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))

        # Softmax + Dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 加权求和: (B, n_head, T, d_head)
        out = torch.matmul(attn, v)

        # 合并多头: (B, n_head, T, d_head) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """
    前馈网络（Position-wise Feed-Forward Network）
    两层 MLP，中间用 GELU 激活函数
    隐藏层维度通常是 d_model 的 4 倍
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer Block = LayerNorm + MultiHeadAttention + 残差连接
                       + LayerNorm + FeedForward + 残差连接

    使用 Pre-LN 架构（先 LayerNorm 再计算），训练更稳定
    """
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        # Pre-LN + 残差连接
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    完整的 GPT 模型：
    1. Token Embedding: 将字符索引映射到向量
    2. Position Embedding: 编码位置信息
    3. N 层 TransformerBlock
    4. LayerNorm + Linear 输出层（预测下一个字符的概率分布）
    """
    def __init__(self, vocab_size, d_model, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        # N 层 Transformer
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_head, dropout) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # 权重共享：token embedding 和输出层共享权重（减少参数量）
        self.lm_head.weight = self.token_emb.weight

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) 输入 token 索引
        targets: (B, T) 目标 token 索引（训练时提供）
        """
        B, T = idx.shape
        assert T <= self.block_size, f"序列长度 {T} 超过 block_size {self.block_size}"

        # Embedding
        tok_emb = self.token_emb(idx)  # (B, T, d_model)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)  # (T, d_model)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)

        # 输出 logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 计算 loss
        loss = None
        if targets is not None:
            # 展平为 (B*T, vocab_size) 和 (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成：每次预测下一个 token，追加到序列末尾

        参数:
        - idx: 初始 token 序列 (B, T)
        - max_new_tokens: 要生成的新 token 数量
        - temperature: 温度参数，越高越随机
        - top_k: 只从概率最高的 k 个 token 中采样
        """
        for _ in range(max_new_tokens):
            # 截断到 block_size（只保留最近的上下文）
            idx_cond = idx[:, -self.block_size:]

            # Forward
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 取最后一个位置的 logits

            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # 采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# ============================================================
# 6. 初始化模型和优化器
# ============================================================
model = GPT(VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数量: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# 初始化实验跟踪器
tracker = ExperimentTracker("day02_char_gpt")
# 记录实验配置信息
import json
config_str = json.dumps({
    "vocab_size": VOCAB_SIZE, "d_model": D_MODEL, "n_head": N_HEAD,
    "n_layer": N_LAYER, "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE,
    "epochs": EPOCHS, "lr": LR, "total_params": total_params,
}, ensure_ascii=False, indent=2)
tracker.log_text("config", config_str)

# ============================================================
# 7. 训练循环
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    """计算验证集 loss"""
    model.eval()
    total_loss = 0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    return total_loss / count


print("\n" + "=" * 60)
print("开始训练字符级 GPT")
print("=" * 60)

for epoch in range(EPOCHS):
    model.train()
    start_time = time.time()
    total_loss = 0
    count = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Forward
        logits, loss = model(x, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

        if (batch_idx + 1) % 200 == 0:
            avg = total_loss / count
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg:.4f}")

    train_loss = total_loss / count
    val_loss = evaluate(model, val_loader, DEVICE)
    elapsed = time.time() - start_time

    # 记录指标
    tracker.log_metric("train_loss", train_loss, step=epoch)
    tracker.log_metric("val_loss", val_loss, step=epoch)

    print(f"\nEpoch {epoch+1}/{EPOCHS} | Time: {elapsed:.0f}s")
    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ============================================================
# 8. 生成文本展示效果
# ============================================================
print("\n" + "=" * 60)
print("生成文本样本")
print("=" * 60)

model.eval()
prompts = ["ROMEO:", "To be or ", "The king"]

for prompt in prompts:
    input_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    output_ids = model.generate(input_ids, max_new_tokens=200, temperature=0.8, top_k=40)
    generated_text = decode(output_ids[0].tolist())
    print(f"\n--- Prompt: '{prompt}' ---")
    print(generated_text)
    print("-" * 40)

    # 保存生成样本到实验记录
    tracker.log_text(f"generation_{prompt[:10]}", f"Prompt: {prompt}\nTemperature: 0.8, Top-k: 40\n\n{generated_text}")

# ============================================================
# 9. 保存实验结果
# ============================================================
print("\n实验摘要:")
print(json.dumps(tracker.summary(), indent=2, ensure_ascii=False))
print("\n训练完成！")
