# 🚀 LLM Training Sprint - 大模型核心能力速通训练平台

> 14 天从 PyTorch 基础到 AgentRL，高密度实战驱动的大模型训练课程

## 适合谁？

有传统开发经验和 ML 基础，想快速补齐 **深度学习训练流程** 和 **后训练技术栈**（SFT → DPO → RL → AgentRL）的 Agent 开发工程师。

## 学习路线全景图

```
                          🎯 14 天学习路线
  ╔══════════════════════════════════════════════════════════════════╗
  ║                                                                ║
  ║   阶段 1: 打地基                阶段 2: 学调教                  ║
  ║   ┌─────────────────────┐      ┌─────────────────────┐        ║
  ║   │ Day 1  PyTorch 基础 │      │ Day 4  SFT 监督微调 │        ║
  ║   │ Day 2  从零写 GPT   │      │ Day 5  数据工程     │        ║
  ║   │ Day 3  HuggingFace  │      └─────────────────────┘        ║
  ║   └─────────────────────┘              │                      ║
  ║            │                           ▼                      ║
  ║            ▼                  阶段 3: 懂对齐                  ║
  ║     会搭模型、会训练           ┌─────────────────────┐        ║
  ║     "做出了一道菜"             │ Day 6  RLHF 原理    │        ║
  ║                                │ Day 7  DPO 实战     │        ║
  ║                                │ Day 8  对齐全景     │        ║
  ║                                └─────────────────────┘        ║
  ║                                        │                      ║
  ║                                        ▼                      ║
  ║                               阶段 4: 玩强化                  ║
  ║                                ┌─────────────────────┐        ║
  ║                                │ Day 9   RL 基础     │        ║
  ║                                │ Day 10  GRPO 训练   │        ║
  ║                                │ Day 11  Agent RL 上 │        ║
  ║                                │ Day 12  Agent RL 下 │        ║
  ║                                └─────────────────────┘        ║
  ║                                        │                      ║
  ║                                        ▼                      ║
  ║                               阶段 5: 大融合                  ║
  ║                                ┌─────────────────────┐        ║
  ║                                │ Day 13-14 期末项目  │        ║
  ║                                │ 端到端训练 Agent     │        ║
  ║                                └─────────────────────┘        ║
  ║                                        │                      ║
  ║                                        ▼                      ║
  ║                                  🎓 毕业！                    ║
  ╚══════════════════════════════════════════════════════════════════╝
```

### 通俗理解：训练大模型就像培养一个员工

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ 预训练   │ ──▶ │ SFT 微调 │ ──▶ │ 偏好对齐 │ ──▶ │ RL 强化  │
  │          │     │          │     │          │     │          │
  │ 读万卷书 │     │ 师傅带徒 │     │ 学规矩   │     │ 实战成长 │
  │ 学知识   │     │ 学格式   │     │ 知好坏   │     │ 越来越强 │
  └──────────┘     └──────────┘     └──────────┘     └──────────┘
   Day 1-3           Day 4-5          Day 6-8         Day 9-14
```

| 阶段 | 天数 | 关键词 | 一句话总结 |
|------|------|--------|-----------|
| 打地基 | Day 1-3 | PyTorch / GPT / HuggingFace | 学会从零搭建和训练一个语言模型 |
| 学调教 | Day 4-5 | SFT / LoRA / 数据质量 | 用高质量数据教模型"说人话" |
| 懂对齐 | Day 6-8 | RLHF / DPO / 对齐技术 | 让模型分清"好回答"和"坏回答" |
| 玩强化 | Day 9-12 | RL / GRPO / AgentRL | 用强化学习让模型自己变强 |
| 大融合 | Day 13-14 | 端到端 Pipeline | 把所有技术串起来训练一个完整 Agent |

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

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    LLM 训练仪表板                           │
  │                                                             │
  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
  │  │ 📈 训练监控 │  │ 🔄 模型对比 │  │ 🗺️ 知识地图      │   │
  │  │             │  │             │  │                  │   │
  │  │ loss 曲线   │  │ Base→SFT    │  │ 14 天进度追踪   │   │
  │  │ 多实验叠加  │  │ →DPO→RL    │  │ 概念检查清单    │   │
  │  │ 实时观测    │  │ 并排对比    │  │ 验证问题        │   │
  │  └─────────────┘  └─────────────┘  └──────────────────┘   │
  └─────────────────────────────────────────────────────────────┘
```

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

> 每个阶段结束后问自己这些问题，能回答说明真正掌握了。

| 阶段 | 验证问题 | 通关标准 |
|------|---------|---------|
| Day 1-3 打地基 | 训练一个模型的完整步骤是什么？ | 能画出 数据→模型→损失→优化 的流程 |
| Day 4-5 学调教 | SFT 的数据需要什么格式？为什么要做 loss masking？ | 能解释指令格式和 masking 的必要性 |
| Day 6-8 懂对齐 | DPO 相比 RLHF 的核心简化是什么？ | 能说出"去掉了奖励模型"及其原理 |
| Day 9-12 玩强化 | GRPO 相比 PPO 省掉了什么？AgentRL 的奖励设计难在哪？ | 能解释组采样和稀疏奖励问题 |
| Day 13-14 大融合 | 从基座到 Agent 模型，经历了哪几个训练阶段？ | 能画出完整 pipeline 并解释每步作用 |

## 推荐资源

| 类型 | 资源 | 适合阶段 |
|------|------|---------|
| 视频 | Andrej Karpathy YouTube 系列、3Blue1Brown Transformer 可视化 | Day 1-3 |
| 课程 | HuggingFace NLP Course | Day 3-5 |
| 书籍 | Sebastian Raschka "Build an LLM from Scratch" | 全程参考 |
| 代码 | `karpathy/nanoGPT`、`huggingface/trl`、`THUDM/AgentTuning` | Day 2, 7, 11 |
| 论文 | Attention Is All You Need、InstructGPT、DPO、LoRA、DeepSeek-R1 | Day 2, 6, 7, 4, 10 |
