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

# 阶段定义（颜色和元数据）
PHASES = [
    {"name": "打地基", "icon": "🧱", "color": "#1976d2", "bg": "#e3f2fd", "days": [1, 2, 3],
     "summary": "从零学会搭建和训练模型，打好深度学习基础"},
    {"name": "学调教", "icon": "🎯", "color": "#388e3c", "bg": "#e8f5e9", "days": [4, 5],
     "summary": "用数据教模型做事，掌握监督微调和数据工程"},
    {"name": "懂对齐", "icon": "⚖️", "color": "#f57c00", "bg": "#fff3e0", "days": [6, 7, 8],
     "summary": "让模型分清好坏，理解偏好对齐技术"},
    {"name": "玩强化", "icon": "🎮", "color": "#c62828", "bg": "#fce4ec", "days": [9, 10, 11, 12],
     "summary": "用强化学习让模型自己变强，训练智能 Agent"},
    {"name": "大融合", "icon": "🏆", "color": "#6a1b9a", "bg": "#f3e5f5", "days": [13, 14],
     "summary": "端到端完成一个 Agent 模型的全流程训练"},
]

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


def _get_phase_for_day(day_num: int) -> dict:
    """根据天数获取所属阶段信息。"""
    for phase in PHASES:
        if day_num in phase["days"]:
            return phase
    return PHASES[0]


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
    # 阶段式进度可视化
    # ------------------------------------------------------------------
    st.markdown("#### 学习路线")

    for phase in PHASES:
        phase_days = [d for d in CURRICULUM if d["day"] in phase["days"]]
        phase_total = sum(len(d["concepts"]) for d in phase_days)
        phase_done = sum(
            1 for d in phase_days for c in d["concepts"]
            if progress["completed_concepts"].get(f"day{d['day']}_{c}", False)
        )
        phase_pct = phase_done / phase_total if phase_total else 0

        # Build day status badges
        day_badges = ""
        for d in phase_days:
            day_num = d["day"]
            day_done = sum(
                1 for c in d["concepts"]
                if progress["completed_concepts"].get(f"day{day_num}_{c}", False)
            )
            day_total = len(d["concepts"])
            ratio = day_done / day_total if day_total else 0
            if ratio == 1:
                badge_bg = "#4caf50"
                badge_text = "white"
                badge_icon = "✓"
            elif ratio > 0:
                badge_bg = "#ff9800"
                badge_text = "white"
                badge_icon = f"{day_done}/{day_total}"
            else:
                badge_bg = "#e0e0e0"
                badge_text = "#999"
                badge_icon = f"0/{day_total}"
            day_badges += (
                f'<span style="display:inline-block; padding:4px 10px; margin:2px 4px; '
                f'border-radius:16px; background:{badge_bg}; color:{badge_text}; '
                f'font-size:12px; font-weight:500;">'
                f'D{day_num} {badge_icon}</span>'
            )

        # Progress bar width
        bar_width = max(phase_pct * 100, 2)
        status_emoji = "✅" if phase_pct == 1 else ("🔨" if phase_pct > 0 else "⏳")

        st.markdown(f"""
        <div style="padding:12px 16px; margin:8px 0; border-radius:10px;
                    background:{phase['bg']}; border-left:4px solid {phase['color']};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:20px;">{phase['icon']}</span>
                    <span style="font-weight:bold; font-size:15px; color:{phase['color']};">
                        {phase['name']}
                    </span>
                    <span style="font-size:12px; color:#888; margin-left:8px;">
                        {phase['summary']}
                    </span>
                </div>
                <div style="font-size:13px;">
                    {status_emoji} {phase_done}/{phase_total}
                </div>
            </div>
            <div style="margin:8px 0 4px 0; background:#00000011; border-radius:4px; height:6px;">
                <div style="width:{bar_width}%; background:{phase['color']}; height:6px;
                            border-radius:4px; transition:width 0.3s;"></div>
            </div>
            <div style="margin-top:6px;">{day_badges}</div>
        </div>
        """, unsafe_allow_html=True)

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
    phase = _get_phase_for_day(day_info["day"])

    # Phase-colored header
    st.markdown(f"""
    <div style="padding:12px 16px; border-radius:8px; background:{phase['bg']};
                border-left:4px solid {phase['color']}; margin-bottom:16px;">
        <span style="font-size:22px;">{phase['icon']}</span>
        <span style="font-weight:bold; font-size:18px; color:{phase['color']};">
            Day {day_info['day']}：{day_info['title']}
        </span>
        <span style="font-size:13px; color:#888; margin-left:12px;">
            {phase['name']}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # 目录检查
    folder_path = PROJECT_ROOT / day_info["folder"]
    if folder_path.exists():
        st.caption(f"📁 对应目录：`{day_info['folder']}/`")
    else:
        st.caption(f"📁 目录 `{day_info['folder']}/` 尚未创建")

    # 概念清单 with progress indicator
    day_concepts_total = len(day_info["concepts"])
    day_concepts_done = sum(
        1 for c in day_info["concepts"]
        if progress["completed_concepts"].get(f"{day_key}_{c}", False)
    )
    st.markdown(f"#### 关键概念 ({day_concepts_done}/{day_concepts_total})")
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
        answer_key = f"{day_key}_q{qi}"
        existing_answer = progress["quiz_answers"].get(answer_key, "")
        has_answer = bool(existing_answer.strip())
        status = "✅" if has_answer else "💭"
        st.markdown(f"{status} **Q{qi+1}:** {question}")
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
