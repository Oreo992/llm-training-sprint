# 🚀 LLM Training Sprint - 大模型核心能力速通训练平台

> 14 天从 PyTorch 基础到 AgentRL，高密度实战驱动的大模型训练课程

## 适合谁？

有传统开发经验和 ML 基础，想快速补齐 **深度学习训练流程** 和 **后训练技术栈**（SFT → DPO → RL → AgentRL）的 Agent 开发工程师。

## 课程路线

```
Day 1-3    深度学习基础        PyTorch 训练循环 → nanoGPT → HuggingFace 生态
Day 4-5    SFT 监督微调        LoRA 微调 → 数据质量实验
Day 6-8    偏好对齐            RLHF 概念 → DPO 实战 → 对齐技术全景
Day 9-12   RL & AgentRL       RL 基础 → GRPO 训练 → Agent 环境 + AgentRL
Day 13-14  综合实战            端到端训练一个 Agent 模型（SFT → DPO → RL）
```

## 项目结构

```
llm-training-sprint/
├── dashboard/                      # Streamlit Web 仪表板
│   ├── app.py                      # 主应用（训练进度、实验对比、知识地图）
│   └── pages/
│       ├── training_monitor.py     # 训练监控（loss/accuracy 曲线）
│       ├── model_compare.py        # 模型对比（多阶段输出并排对比）
│       └── knowledge_map.py        # 知识地图（学习进度追踪）
├── utils/
│   ├── experiment_tracker.py       # 实验跟踪器（自动记录指标到 JSON）
│   └── model_comparator.py        # 模型对比工具（perplexity 等指标）
├── day01_pytorch_basics/
│   └── train_imdb_classifier.py    # 手写 Transformer + training loop, IMDb 分类
├── day02_nanoGPT/
│   └── train_char_gpt.py          # 从零实现 GPT，Shakespeare 字符级训练
├── day03_huggingface/
│   └── hf_exploration.py          # Qwen2-0.5B 加载、tokenizer 分析、HF Trainer
├── day04_sft/
│   └── sft_training.py            # SFTTrainer + LoRA，中文指令微调
├── day05_data_engineering/
│   └── data_quality_experiment.py  # 高质量 vs 低质量数据对比实验
├── day06_rlhf_concepts/
│   └── rlhf_visual_guide.py       # RLHF 流程可视化
├── day07_dpo/
│   └── dpo_training.py            # DPOTrainer 实战，SFT vs DPO 对比
├── day08_alignment_landscape/
│   └── alignment_overview.py      # 7 种对齐技术演进图
├── day09_rl_basics/
│   └── rl_concepts.py             # REINFORCE 算法 + RL→LLM 概念映射
├── day10_grpo/
│   └── grpo_training.py           # GRPO 在 GSM8K 上训练（DeepSeek-R1 式）
├── day11_12_agent_rl/
│   ├── agent_env.py               # Agent 环境（文件操作任务）
│   └── agent_rl_training.py       # AgentRL 训练
├── day13_14_final_project/
│   ├── agent_data_generator.py    # Agent 训练数据生成器
│   ├── end_to_end_agent_training.py  # Base → SFT → DPO → RL 完整 pipeline
│   └── evaluation.py              # 多阶段模型评估 + 可视化报告
└── requirements.txt
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Web 仪表板
streamlit run dashboard/app.py

# 3. 按天运行训练脚本
python day01_pytorch_basics/train_imdb_classifier.py
```

## 核心特性

### 📊 Web 仪表板
- **训练监控**：实时 loss/accuracy 曲线，多实验叠加对比
- **模型对比**：输入 prompt，并排查看 Base / SFT / DPO / RL 各阶段模型输出
- **知识地图**：14 天学习进度追踪，概念检查清单

### 🔬 实验跟踪
所有训练脚本自动集成 `ExperimentTracker`，指标保存到 `experiments/` 目录：

```python
from utils.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("my_experiment")
tracker.log_metric("loss", 0.5, step=100)
tracker.log_text("sample_output", model_response)
```

### 🔄 无 GPU 降级
所有脚本在无 GPU 环境下自动切换到模拟演示模式，确保可以在任何机器上跑通流程。

## 每日验证问题

| 模块 | 验证问题 |
|------|---------|
| Day 1-3 | 训练一个模型的完整步骤是什么？ |
| Day 4-5 | SFT 的数据需要什么格式？为什么要做 loss masking？ |
| Day 6-8 | DPO 相比 RLHF 的核心简化是什么？ |
| Day 9-12 | GRPO 相比 PPO 省掉了什么？AgentRL 的奖励设计难在哪？ |
| Day 13-14 | 从基座到 Agent 模型，经历了哪几个训练阶段？ |

## 推荐资源

| 类型 | 资源 |
|------|------|
| 视频 | Andrej Karpathy YouTube 系列、3Blue1Brown Transformer 可视化 |
| 课程 | HuggingFace NLP Course |
| 书籍 | Sebastian Raschka "Build an LLM from Scratch" |
| 代码 | `karpathy/nanoGPT`、`huggingface/trl`、`THUDM/AgentTuning` |
| 论文 | Attention Is All You Need、InstructGPT、DPO、LoRA、DeepSeek-R1 |
