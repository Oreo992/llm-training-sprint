"""
端到端 Agent 模型训练 Pipeline
==============================

完整训练流程：Base Model → SFT → DPO → RL（可选 GRPO）

核心概念：
1. SFT（Supervised Fine-Tuning）：用高质量 Agent 对话数据微调基座模型，
   教会模型 function calling 的格式和基本的工具使用能力。

2. DPO（Direct Preference Optimization）：用好/坏 Agent 轨迹做偏好对齐，
   无需奖励模型，直接优化策略使其偏好好的轨迹。
   损失函数：L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

3. GRPO（Group Relative Policy Optimization）：在模拟环境中用强化学习继续优化，
   通过组内相对排名计算优势值，避免需要 critic 网络。

基座模型：Qwen2-0.5B（小巧高效，适合教学演示）
"""

import json
import os
import sys
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 处理 import 路径，确保能从项目根目录导入 utils
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.experiment_tracker import ExperimentTracker

# 导入本地数据生成器
from agent_data_generator import generate_sft_data, generate_dpo_data

# ============================================================================
# 配置
# ============================================================================

@dataclass
class TrainingConfig:
    """训练超参数配置"""
    # 模型
    model_name: str = "Qwen/Qwen2-0.5B"

    # SFT 阶段
    sft_epochs: int = 3
    sft_lr: float = 2e-5
    sft_batch_size: int = 2
    sft_max_length: int = 512

    # DPO 阶段
    dpo_epochs: int = 2
    dpo_lr: float = 5e-6
    dpo_batch_size: int = 2
    dpo_beta: float = 0.1  # DPO 温度系数，控制偏离参考模型的程度
    dpo_max_length: int = 512

    # GRPO 阶段（可选）
    grpo_epochs: int = 1
    grpo_lr: float = 1e-6
    grpo_batch_size: int = 4
    grpo_group_size: int = 4  # 每个 prompt 采样的回复数
    grpo_clip_eps: float = 0.2  # PPO-clip 范围
    grpo_kl_coeff: float = 0.05  # KL 惩罚系数

    # 通用
    output_dir: str = "checkpoints"
    seed: int = 42
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    use_fp16: bool = True
    device: str = "auto"

    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ============================================================================
# 数据集类
# ============================================================================

class AgentSFTDataset(Dataset):
    """
    SFT 数据集：将多轮对话转换为模型可训练的 token 序列。

    关键设计：
    - 只对 assistant 回复部分计算 loss（因为我们只想让模型学会如何回复）
    - system/user/tool 消息部分的 label 设为 -100（被 CrossEntropy 忽略）
    """
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]

        # 将多轮对话拼接为文本（简化处理，实际应使用 chat template）
        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "assistant" and msg.get("tool_calls"):
                # 将 tool_calls 序列化为文本
                tc = msg["tool_calls"][0]["function"]
                content = f"<tool_call>{json.dumps({'name': tc['name'], 'arguments': json.loads(tc['arguments'])}, ensure_ascii=False)}</tool_call>"
            if content:
                text_parts.append(f"<|{role}|>\n{content}")

        text = "\n".join(text_parts) + self.tokenizer.eos_token

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        # SFT: label = input_ids（自回归训练，简化版本对所有 token 计算 loss）
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # padding 部分不计算 loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class AgentDPODataset(Dataset):
    """
    DPO 数据集：每条数据包含同一个 prompt 的 chosen 和 rejected 回复。

    DPO 的核心思想：
    - 不需要显式的奖励模型
    - 直接从偏好对中学习：使 chosen 的概率相对于 rejected 更高
    - β 参数控制偏离参考模型的程度（β 越大越保守）
    """
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _messages_to_text(self, messages: List[Dict]) -> str:
        """将消息列表转为文本"""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "assistant" and msg.get("tool_calls"):
                tc = msg["tool_calls"][0]["function"]
                content = f"<tool_call>{json.dumps({'name': tc['name'], 'arguments': json.loads(tc['arguments'])}, ensure_ascii=False)}</tool_call>"
            if content:
                parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt_text = self._messages_to_text(item["prompt"])
        chosen_text = prompt_text + "\n" + self._messages_to_text(item["chosen"]) + self.tokenizer.eos_token
        rejected_text = prompt_text + "\n" + self._messages_to_text(item["rejected"]) + self.tokenizer.eos_token

        chosen_enc = self.tokenizer(chosen_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        rejected_enc = self.tokenizer(rejected_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


# ============================================================================
# Stage 1: SFT 训练
# ============================================================================

def train_sft(
    model,
    tokenizer,
    config: TrainingConfig,
    tracker: ExperimentTracker,
) -> None:
    """
    Stage 1: 有监督微调（SFT）

    目标：让基座模型学会 Agent 对话格式，特别是 function calling。

    训练策略：
    - 标准的自回归语言模型训练（next token prediction）
    - 只对 assistant 回复部分计算 loss（简化版本对所有非 padding token 计算）
    - 使用较小的学习率（2e-5）避免灾难性遗忘
    """
    print("\n" + "=" * 60)
    print("Stage 1: SFT（有监督微调）")
    print("=" * 60)

    device = config.get_device()
    model = model.to(device)
    model.train()

    # 生成训练数据
    sft_data = generate_sft_data(num_samples=50, seed=config.seed)
    dataset = AgentSFTDataset(sft_data, tokenizer, config.sft_max_length)
    dataloader = DataLoader(dataset, batch_size=config.sft_batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.sft_lr, weight_decay=0.01)

    # 学习率调度器：线性 warmup + cosine decay
    total_steps = len(dataloader) * config.sft_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.sft_lr, total_steps=total_steps, pct_start=0.1
    )

    global_step = 0
    for epoch in range(config.sft_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += outputs.loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                current_loss = outputs.loss.item()
                print(f"  Epoch {epoch+1}/{config.sft_epochs}, "
                      f"Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {current_loss:.4f}")
                tracker.log_metric("sft/loss", current_loss, step=global_step)
                tracker.log_metric("sft/lr", scheduler.get_last_lr()[0], step=global_step)

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1} 平均 Loss: {avg_loss:.4f}")
        tracker.log_metric("sft/epoch_loss", avg_loss, step=epoch)

    # 保存 SFT 检查点
    sft_dir = os.path.join(config.output_dir, "sft")
    os.makedirs(sft_dir, exist_ok=True)
    model.save_pretrained(sft_dir)
    tokenizer.save_pretrained(sft_dir)
    print(f"  SFT 模型已保存到 {sft_dir}")
    tracker.log_metric("sft/final_loss", avg_loss)

    return model


# ============================================================================
# Stage 2: DPO 训练
# ============================================================================

def compute_dpo_loss(
    model,
    ref_model,
    chosen_input_ids,
    chosen_attention_mask,
    rejected_input_ids,
    rejected_attention_mask,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算 DPO 损失。

    DPO 损失公式：
    L = -log σ(β * (log π_θ(y_w|x) / π_ref(y_w|x) - log π_θ(y_l|x) / π_ref(y_l|x)))

    其中：
    - π_θ 是当前策略（正在训练的模型）
    - π_ref 是参考策略（冻结的 SFT 模型）
    - y_w 是 chosen（好的回复），y_l 是 rejected（差的回复）
    - β 控制偏离参考模型的程度

    直觉：让好回复相对于参考模型的概率比提升，坏回复的比降低。
    """
    # 计算当前模型的 log probabilities
    def get_log_probs(model, input_ids, attention_mask):
        with torch.no_grad() if not model.training else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]  # 去掉最后一个 token 的 logits
            labels = input_ids[:, 1:]  # 目标是下一个 token
            log_probs = F.log_softmax(logits, dim=-1)
            # 取对应 label 的 log prob
            token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
            # 用 attention_mask 过滤 padding
            mask = attention_mask[:, 1:]
            return (token_log_probs * mask).sum(dim=-1)

    # 当前策略的 log prob
    policy_chosen_logps = get_log_probs(model, chosen_input_ids, chosen_attention_mask)
    policy_rejected_logps = get_log_probs(model, rejected_input_ids, rejected_attention_mask)

    # 参考策略的 log prob（不计算梯度）
    with torch.no_grad():
        ref_chosen_logps = get_log_probs(ref_model, chosen_input_ids, chosen_attention_mask)
        ref_rejected_logps = get_log_probs(ref_model, rejected_input_ids, rejected_attention_mask)

    # DPO 损失
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # L = -log σ(chosen_rewards - rejected_rewards)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    # 记录指标
    metrics = {
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
    }

    return loss, metrics


def train_dpo(
    model,
    ref_model,
    tokenizer,
    config: TrainingConfig,
    tracker: ExperimentTracker,
) -> None:
    """
    Stage 2: DPO（直接偏好优化）

    目标：通过对比好/坏 Agent 轨迹，让模型学会偏好更好的行为模式。

    与 RLHF 的区别：
    - RLHF：先训练奖励模型 → 再用 PPO 优化策略
    - DPO：直接从偏好数据优化策略，跳过奖励模型训练

    关键参数：
    - β（beta）：控制偏离参考模型的程度
      - β 太小：模型变化太大，可能丢失基础能力
      - β 太大：模型几乎不变，学不到偏好
    """
    print("\n" + "=" * 60)
    print("Stage 2: DPO（直接偏好优化）")
    print("=" * 60)

    device = config.get_device()
    model = model.to(device)
    ref_model = ref_model.to(device)
    ref_model.eval()  # 参考模型始终保持 eval 模式

    model.train()

    # 生成 DPO 数据
    dpo_data = generate_dpo_data(num_samples=30, seed=config.seed)
    dataset = AgentDPODataset(dpo_data, tokenizer, config.dpo_max_length)
    dataloader = DataLoader(dataset, batch_size=config.dpo_batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.dpo_lr, weight_decay=0.01)

    global_step = 0
    for epoch in range(config.dpo_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)

            loss, metrics = compute_dpo_loss(
                model, ref_model,
                chosen_input_ids, chosen_attention_mask,
                rejected_input_ids, rejected_attention_mask,
                beta=config.dpo_beta,
            )

            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item()
            epoch_accuracy += metrics["accuracy"]
            num_batches += 1

            if batch_idx % 5 == 0:
                print(f"  Epoch {epoch+1}/{config.dpo_epochs}, "
                      f"Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Accuracy: {metrics['accuracy']:.2%}, "
                      f"Reward Margin: {metrics['reward_margin']:.4f}")
                tracker.log_metric("dpo/loss", loss.item(), step=global_step)
                tracker.log_metric("dpo/accuracy", metrics["accuracy"], step=global_step)
                tracker.log_metric("dpo/reward_margin", metrics["reward_margin"], step=global_step)

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_accuracy / max(num_batches, 1)
        print(f"  Epoch {epoch+1} 平均 Loss: {avg_loss:.4f}, 平均 Accuracy: {avg_acc:.2%}")
        tracker.log_metric("dpo/epoch_loss", avg_loss, step=epoch)
        tracker.log_metric("dpo/epoch_accuracy", avg_acc, step=epoch)

    # 保存 DPO 检查点
    dpo_dir = os.path.join(config.output_dir, "dpo")
    os.makedirs(dpo_dir, exist_ok=True)
    model.save_pretrained(dpo_dir)
    tokenizer.save_pretrained(dpo_dir)
    print(f"  DPO 模型已保存到 {dpo_dir}")
    tracker.log_metric("dpo/final_loss", avg_loss)
    tracker.log_metric("dpo/final_accuracy", avg_acc)

    return model


# ============================================================================
# Stage 3: GRPO 训练（可选）
# ============================================================================

def simulate_environment_reward(response: str, task_query: str) -> float:
    """
    模拟环境奖励函数。

    在真实场景中，这里会是一个实际的环境（如 API 沙箱），
    模型调用工具后环境返回结果，根据任务完成度给出奖励。

    奖励信号设计：
    - 格式正确（使用了 <tool_call> 标签）：+0.3
    - 工具选择合理（tool name 与 query 相关）：+0.3
    - 回答完整（长度适中、有结构）：+0.2
    - 没有胡编乱造（不含明显的幻觉标记）：+0.2
    """
    reward = 0.0

    # 格式奖励：是否使用了工具调用格式
    if "<tool_call>" in response:
        reward += 0.3

    # 回答质量（简化：基于长度和结构化特征）
    if len(response) > 50:
        reward += 0.1
    if len(response) > 150:
        reward += 0.1
    if "\n" in response:  # 有换行表示有结构
        reward += 0.1

    # 惩罚幻觉
    hallucination_markers = ["500%", "完全取代", "编造", "虚构"]
    if any(m in response for m in hallucination_markers):
        reward -= 0.3

    # 任务相关性（简化检查）
    if any(keyword in response for keyword in ["搜索", "计算", "天气", "文件", "结果"]):
        reward += 0.2

    return max(0.0, min(1.0, reward))  # clip 到 [0, 1]


def train_grpo(
    model,
    tokenizer,
    config: TrainingConfig,
    tracker: ExperimentTracker,
) -> None:
    """
    Stage 3: GRPO（Group Relative Policy Optimization）— 可选

    GRPO 的核心思想：
    1. 对每个 prompt，采样 G 个回复（group_size）
    2. 用环境奖励函数对每个回复评分
    3. 在组内计算相对优势值：A_i = (r_i - mean(r)) / std(r)
    4. 用优势值加权更新策略，同时施加 KL 约束

    与 PPO 的区别：
    - PPO 需要一个 critic 网络来估计价值函数
    - GRPO 通过组内相对排名替代 critic，更简洁高效

    注意：这里是简化演示版本，实际 GRPO 需要更复杂的采样和梯度计算。
    """
    print("\n" + "=" * 60)
    print("Stage 3: GRPO（组相对策略优化）[可选]")
    print("=" * 60)

    device = config.get_device()
    model = model.to(device)

    # 准备一组用于 RL 训练的 prompt
    prompts = [
        "帮我搜索一下大语言模型的最新进展，并总结要点。",
        "请计算 (15 + 27) * 3 - 18 / 6，并解释计算过程。",
        "查看北京的天气，帮我规划明天的出行方案。",
        "搜索中国GDP增长率相关信息，计算关键数据，并将报告保存到 report.txt。",
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.grpo_lr, weight_decay=0.01)

    # 保存初始模型参数，用于计算 KL 散度
    ref_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    for epoch in range(config.grpo_epochs):
        epoch_reward = 0.0
        num_prompts = 0

        for prompt in prompts:
            # 对每个 prompt 采样 G 个回复
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            responses = []
            rewards = []

            model.eval()
            with torch.no_grad():
                for _ in range(config.grpo_group_size):
                    output = model.generate(
                        input_ids,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    response_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
                    responses.append(response_text)
                    reward = simulate_environment_reward(response_text, prompt)
                    rewards.append(reward)

            model.train()

            # 计算组内相对优势（GRPO 核心）
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            mean_reward = rewards_tensor.mean()
            std_reward = rewards_tensor.std() + 1e-8
            advantages = (rewards_tensor - mean_reward) / std_reward  # 标准化优势

            # 用优势值加权的策略梯度更新
            # 简化：选择优势最高的回复做梯度更新（类似 best-of-n + fine-tune）
            best_idx = advantages.argmax().item()
            best_response = responses[best_idx]

            if rewards[best_idx] > 0.3:  # 只有奖励足够高才更新
                full_text = prompt + best_response + tokenizer.eos_token
                encoding = tokenizer(full_text, return_tensors="pt", max_length=config.dpo_max_length, truncation=True).to(device)

                outputs = model(**encoding, labels=encoding["input_ids"])
                loss = outputs.loss * advantages[best_idx].item()  # 优势加权

                # KL 惩罚：防止策略偏离太远
                kl_penalty = 0.0
                for name, param in model.named_parameters():
                    if name in ref_params:
                        kl_penalty += ((param - ref_params[name]) ** 2).sum()
                kl_penalty = config.grpo_kl_coeff * kl_penalty

                total_loss = loss + kl_penalty
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            epoch_reward += mean_reward.item()
            num_prompts += 1

            print(f"  Prompt: {prompt[:30]}... | "
                  f"Mean Reward: {mean_reward:.3f} | "
                  f"Best Reward: {max(rewards):.3f}")
            tracker.log_metric("grpo/mean_reward", mean_reward.item())
            tracker.log_metric("grpo/best_reward", max(rewards))

        avg_reward = epoch_reward / max(num_prompts, 1)
        print(f"  Epoch {epoch+1} 平均 Reward: {avg_reward:.4f}")
        tracker.log_metric("grpo/epoch_reward", avg_reward, step=epoch)

    # 保存 GRPO 检查点
    grpo_dir = os.path.join(config.output_dir, "grpo")
    os.makedirs(grpo_dir, exist_ok=True)
    model.save_pretrained(grpo_dir)
    tokenizer.save_pretrained(grpo_dir)
    print(f"  GRPO 模型已保存到 {grpo_dir}")
    tracker.log_metric("grpo/final_reward", avg_reward)

    return model


# ============================================================================
# 主 Pipeline
# ============================================================================

def run_pipeline(enable_grpo: bool = False):
    """
    执行完整的端到端训练 Pipeline。

    流程：
    1. 加载基座模型（Qwen2-0.5B）
    2. Stage 1: SFT — 学习 function calling 格式
    3. Stage 2: DPO — 学习偏好好的 Agent 轨迹
    4. Stage 3: GRPO（可选）— 在模拟环境中强化学习
    """
    config = TrainingConfig()
    device = config.get_device()
    print(f"使用设备: {device}")

    # 初始化实验跟踪器
    tracker = ExperimentTracker(
        experiment_name="day13_14_agent_training",
        tags={"day": "13-14", "type": "end_to_end"},
    )

    # ------------------------------------------------------------------
    # 加载基座模型
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"加载基座模型: {config.model_name}")
    print("=" * 60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.use_fp16 and device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保已安装 transformers 并可以访问 Qwen2-0.5B 模型。")
        print("安装命令: pip install transformers accelerate")
        return

    # ------------------------------------------------------------------
    # Stage 1: SFT
    # ------------------------------------------------------------------
    model = train_sft(model, tokenizer, config, tracker)

    # ------------------------------------------------------------------
    # Stage 2: DPO
    # ------------------------------------------------------------------
    # 保存一份 SFT 后的模型作为 DPO 的参考模型（ref_model）
    print("\n准备 DPO 参考模型（冻结 SFT 模型副本）...")
    ref_model = copy.deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False

    model = train_dpo(model, ref_model, tokenizer, config, tracker)

    # 释放参考模型显存
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage 3: GRPO（可选）
    # ------------------------------------------------------------------
    if enable_grpo:
        model = train_grpo(model, tokenizer, config, tracker)

    # ------------------------------------------------------------------
    # 训练完成
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("训练 Pipeline 完成！")
    print("=" * 60)
    print(f"\n实验摘要:")
    summary = tracker.summary()
    for key, val in summary.items():
        if isinstance(val, dict):
            print(f"  {key}: latest={val['latest']:.4f}, min={val['min']:.4f}, max={val['max']:.4f}")

    print(f"\n检查点保存目录: {config.output_dir}")
    print(f"  - SFT:  {os.path.join(config.output_dir, 'sft')}")
    print(f"  - DPO:  {os.path.join(config.output_dir, 'dpo')}")
    if enable_grpo:
        print(f"  - GRPO: {os.path.join(config.output_dir, 'grpo')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="端到端 Agent 模型训练")
    parser.add_argument("--enable-grpo", action="store_true", help="启用 GRPO 阶段")
    args = parser.parse_args()

    run_pipeline(enable_grpo=args.enable_grpo)
