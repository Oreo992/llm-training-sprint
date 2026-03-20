"""
知识地图页面 - 展示 14 天学习路线的可视化进度、概念检查和笔记。
"""

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 用户进度保存路径
PROGRESS_FILE = PROJECT_ROOT / "experiments" / "user_progress.json"

# 14 天课程大纲
CURRICULUM = [
    {
        "day": 1,
        "title": "PyTorch 基础",
        "folder": "day01_pytorch_basics",
        "concepts": ["Tensor 操作", "自动微分", "nn.Module", "DataLoader", "损失函数与优化器"],
        "quiz": [
            "torch.Tensor 和 numpy.ndarray 的主要区别是什么？",
            "什么是计算图？PyTorch 如何实现自动求导？",
        ],
    },
    {
        "day": 2,
        "title": "nanoGPT 实战",
        "folder": "day02_nanoGPT",
        "concepts": ["Transformer 架构", "注意力机制", "位置编码", "因果语言建模", "BPE 分词"],
        "quiz": [
            "Self-Attention 的时间复杂度是多少？为什么？",
            "因果注意力掩码的作用是什么？",
        ],
    },
    {
        "day": 3,
        "title": "HuggingFace 生态",
        "folder": "day03_huggingface",
        "concepts": ["Transformers 库", "Tokenizer", "Pipeline", "模型加载与保存", "Datasets 库"],
        "quiz": [
            "AutoModelForCausalLM 和 AutoModelForSeq2SeqLM 的区别？",
            "如何使用 LoRA 减少微调参数量？",
        ],
    },
    {
        "day": 4,
        "title": "监督微调 (SFT)",
        "folder": "day04_sft",
        "concepts": ["指令微调", "数据格式化", "LoRA / QLoRA", "训练超参数", "过拟合检测"],
        "quiz": [
            "SFT 中 instruction、input、output 三个字段的作用？",
            "LoRA 的秩 r 如何影响训练效果？",
        ],
    },
    {
        "day": 5,
        "title": "数据工程",
        "folder": "day05_data_engineering",
        "concepts": ["数据清洗", "数据去重", "质量过滤", "数据配比", "合成数据"],
        "quiz": [
            "为什么数据质量比数据数量更重要？",
            "MinHash 去重的基本原理是什么？",
        ],
    },
    {
        "day": 6,
        "title": "RLHF 概念",
        "folder": "day06_rlhf_concepts",
        "concepts": ["奖励模型", "人类偏好", "Bradley-Terry 模型", "PPO 算法", "KL 散度约束"],
        "quiz": [
            "RLHF 流程的三个阶段是什么？",
            "为什么需要 KL 散度惩罚？",
        ],
    },
    {
        "day": 7,
        "title": "DPO 直接偏好优化",
        "folder": "day07_dpo",
        "concepts": ["DPO 损失函数", "偏好数据构建", "隐式奖励", "参考模型", "β 参数调节"],
        "quiz": [
            "DPO 相比 RLHF 的主要优势是什么？",
            "DPO 的 β 参数控制什么？",
        ],
    },
    {
        "day": 8,
        "title": "对齐全景",
        "folder": "day08_alignment_landscape",
        "concepts": ["ORPO", "KTO", "SimPO", "IPO", "对齐税"],
        "quiz": [
            "对齐方法的核心目标是什么？",
            "什么是对齐税（alignment tax）？",
        ],
    },
    {
        "day": 9,
        "title": "RL 基础",
        "folder": "day09_rl_basics",
        "concepts": ["MDP", "策略梯度", "价值函数", "Actor-Critic", "GAE"],
        "quiz": [
            "策略梯度定理的直觉解释？",
            "GAE 的 λ 参数如何平衡偏差和方差？",
        ],
    },
    {
        "day": 10,
        "title": "GRPO",
        "folder": "day10_grpo",
        "concepts": ["Group Relative Policy Optimization", "无 Critic 设计", "组采样", "优势估计", "DeepSeek 实践"],
        "quiz": [
            "GRPO 如何避免训练 Critic 网络？",
            "组内相对奖励的计算方式？",
        ],
    },
    {
        "day": 11,
        "title": "Agent RL（上）",
        "folder": "day11_12_agent_rl",
        "concepts": ["工具调用", "ReAct 范式", "环境交互", "轨迹奖励", "Multi-turn RL"],
        "quiz": [
            "Agent RL 和传统 RLHF 的关键区别？",
            "如何为工具调用设计奖励函数？",
        ],
    },
    {
        "day": 12,
        "title": "Agent RL（下）",
        "folder": "day11_12_agent_rl",
        "concepts": ["代码生成奖励", "沙箱执行", "多步推理", "过程奖励模型", "课程学习"],
        "quiz": [
            "过程奖励模型（PRM）和结果奖励模型（ORM）的区别？",
            "课程学习在 Agent RL 中如何应用？",
        ],
    },
    {
        "day": 13,
        "title": "期末项目（上）",
        "folder": "day13_14_final_project",
        "concepts": ["项目规划", "数据收集", "模型选择", "训练流水线", "评估策略"],
        "quiz": [
            "如何设计一个端到端的 LLM 训练项目？",
        ],
    },
    {
        "day": 14,
        "title": "期末项目（下）",
        "folder": "day13_14_final_project",
        "concepts": ["模型评估", "消融实验", "结果分析", "项目总结", "后续方向"],
        "quiz": [
            "如何进行有意义的消融实验？",
        ],
    },
]


def _load_progress() -> dict:
    """加载用户进度。"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_concepts": {}, "notes": {}, "quiz_answers": {}}


def _save_progress(progress: dict) -> None:
    """保存用户进度。"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def render():
    """渲染知识地图页面。"""
    st.header("🗺️ 知识地图")

    progress = _load_progress()

    # ------------------------------------------------------------------
    # 总体进度
    # ------------------------------------------------------------------
    total_concepts = sum(len(d["concepts"]) for d in CURRICULUM)
    completed = sum(
        1
        for d in CURRICULUM
        for c in d["concepts"]
        if progress["completed_concepts"].get(f"day{d['day']}_{c}", False)
    )
    pct = completed / total_concepts if total_concepts else 0

    st.progress(pct, text=f"总体进度：{completed}/{total_concepts} 概念已掌握 ({pct:.0%})")

    # ------------------------------------------------------------------
    # 进度条网格
    # ------------------------------------------------------------------
    cols = st.columns(7)
    for i, day_info in enumerate(CURRICULUM):
        day_num = day_info["day"]
        day_completed = sum(
            1 for c in day_info["concepts"]
            if progress["completed_concepts"].get(f"day{day_num}_{c}", False)
        )
        day_total = len(day_info["concepts"])
        with cols[i % 7]:
            ratio = day_completed / day_total if day_total else 0
            color = "🟢" if ratio == 1 else ("🟡" if ratio > 0 else "⚪")
            st.markdown(f"{color} **Day {day_num}**  \n{day_completed}/{day_total}")

    st.divider()

    # ------------------------------------------------------------------
    # 每日详情
    # ------------------------------------------------------------------
    selected_day = st.selectbox(
        "选择天数查看详情",
        range(1, 15),
        format_func=lambda d: f"Day {d} - {CURRICULUM[d-1]['title']}",
    )

    day_info = CURRICULUM[selected_day - 1]
    day_key = f"day{day_info['day']}"

    st.subheader(f"Day {day_info['day']}：{day_info['title']}")

    # 目录检查
    folder_path = PROJECT_ROOT / day_info["folder"]
    if folder_path.exists():
        st.caption(f"📁 对应目录：`{day_info['folder']}/`")
    else:
        st.caption(f"📁 目录 `{day_info['folder']}/` 尚未创建")

    # 概念清单
    st.markdown("#### 关键概念")
    changed = False
    for concept in day_info["concepts"]:
        key = f"{day_key}_{concept}"
        current = progress["completed_concepts"].get(key, False)
        new_val = st.checkbox(concept, value=current, key=f"chk_{key}")
        if new_val != current:
            progress["completed_concepts"][key] = new_val
            changed = True

    # 验证问题
    st.markdown("#### 验证问题")
    for qi, question in enumerate(day_info["quiz"]):
        st.markdown(f"**Q{qi+1}:** {question}")
        answer_key = f"{day_key}_q{qi}"
        existing_answer = progress["quiz_answers"].get(answer_key, "")
        answer = st.text_area(
            f"你的回答", value=existing_answer, key=f"ans_{answer_key}", height=80,
            label_visibility="collapsed",
        )
        if answer != existing_answer:
            progress["quiz_answers"][answer_key] = answer
            changed = True

    # 用户笔记
    st.markdown("#### 学习笔记")
    note_key = f"{day_key}_notes"
    existing_note = progress["notes"].get(note_key, "")
    note = st.text_area("记录笔记", value=existing_note, key=f"note_{note_key}", height=150)
    if note != existing_note:
        progress["notes"][note_key] = note
        changed = True

    # 保存
    if changed:
        _save_progress(progress)
        st.toast("进度已自动保存！")
