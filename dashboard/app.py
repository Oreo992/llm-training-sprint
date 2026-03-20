"""
LLM 训练课程仪表板 - 主应用入口。

启动方式：
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------
# 页面配置
# ------------------------------------------------------------------
st.set_page_config(
    page_title="LLM 训练实验仪表板",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# 总览页面
# ------------------------------------------------------------------

def _render_overview():
    """训练进度总览页面。"""
    st.header("训练进度总览")

    from utils.experiment_tracker import ExperimentTracker

    experiments_dir = PROJECT_ROOT / "experiments"
    exp_names = ExperimentTracker.list_experiments(str(experiments_dir))

    # ------------------------------------------------------------------
    # 训练 Pipeline 可视化
    # ------------------------------------------------------------------
    st.subheader("训练流程全景")
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; gap: 8px; flex-wrap: wrap;
                padding: 20px; background: linear-gradient(135deg, #667eea11, #764ba211); border-radius: 12px; margin-bottom: 20px;">
        <div style="text-align:center; padding:16px 20px; background:#e3f2fd; border-radius:10px; min-width:130px; border-left:4px solid #1976d2;">
            <div style="font-size:28px;">📚</div>
            <div style="font-weight:bold; color:#1976d2;">预训练</div>
            <div style="font-size:12px; color:#666;">读万卷书<br/>学知识</div>
        </div>
        <div style="font-size:24px; color:#999;">➜</div>
        <div style="text-align:center; padding:16px 20px; background:#e8f5e9; border-radius:10px; min-width:130px; border-left:4px solid #388e3c;">
            <div style="font-size:28px;">🎓</div>
            <div style="font-weight:bold; color:#388e3c;">SFT 微调</div>
            <div style="font-size:12px; color:#666;">师傅带徒<br/>学格式</div>
        </div>
        <div style="font-size:24px; color:#999;">➜</div>
        <div style="text-align:center; padding:16px 20px; background:#fff3e0; border-radius:10px; min-width:130px; border-left:4px solid #f57c00;">
            <div style="font-size:28px;">⚖️</div>
            <div style="font-weight:bold; color:#f57c00;">偏好对齐</div>
            <div style="font-size:12px; color:#666;">学规矩<br/>知好坏</div>
        </div>
        <div style="font-size:24px; color:#999;">➜</div>
        <div style="text-align:center; padding:16px 20px; background:#fce4ec; border-radius:10px; min-width:130px; border-left:4px solid #c62828;">
            <div style="font-size:28px;">🚀</div>
            <div style="font-weight:bold; color:#c62828;">RL 强化</div>
            <div style="font-size:12px; color:#666;">实战成长<br/>越来越强</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 状态统计卡片
    # ------------------------------------------------------------------
    day_folders = [
        "day01_pytorch_basics", "day02_nanoGPT", "day03_huggingface",
        "day04_sft", "day05_data_engineering", "day06_rlhf_concepts",
        "day07_dpo", "day08_alignment_landscape", "day09_rl_basics",
        "day10_grpo", "day11_12_agent_rl", "day13_14_final_project",
    ]
    existing = sum(1 for f in day_folders if (PROJECT_ROOT / f).exists())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("实验总数", len(exp_names))
    col2.metric("课程目录", f"{existing}/{len(day_folders)}")
    col3.metric("训练阶段", "4 个阶段")
    col4.metric("总天数", "14 天")

    st.divider()

    # ------------------------------------------------------------------
    # 课程阶段进度
    # ------------------------------------------------------------------
    st.subheader("课程阶段")
    phases = [
        {"name": "阶段 1：打地基", "icon": "🧱", "days": "Day 1-3", "color": "#1976d2",
         "desc": "PyTorch → nanoGPT → HuggingFace", "folders": day_folders[:3]},
        {"name": "阶段 2：学调教", "icon": "🎯", "days": "Day 4-5", "color": "#388e3c",
         "desc": "SFT 监督微调 → 数据质量实验", "folders": day_folders[3:5]},
        {"name": "阶段 3：懂对齐", "icon": "⚖️", "days": "Day 6-8", "color": "#f57c00",
         "desc": "RLHF → DPO → 对齐全景", "folders": day_folders[5:8]},
        {"name": "阶段 4：玩强化", "icon": "🎮", "days": "Day 9-12", "color": "#c62828",
         "desc": "RL 基础 → GRPO → AgentRL", "folders": day_folders[8:10] + [day_folders[10]]},
        {"name": "阶段 5：大融合", "icon": "🏆", "days": "Day 13-14", "color": "#6a1b9a",
         "desc": "端到端 Agent 训练", "folders": [day_folders[11]]},
    ]

    phase_cols = st.columns(len(phases))
    for i, phase in enumerate(phases):
        with phase_cols[i]:
            done = sum(1 for f in phase["folders"] if (PROJECT_ROOT / f).exists())
            total = len(phase["folders"])
            pct = done / total if total else 0
            status = "✅" if pct == 1 else ("🔨" if pct > 0 else "⏳")
            st.markdown(f"""
            <div style="text-align:center; padding:12px; border-radius:10px;
                        background: {phase['color']}11; border-top: 3px solid {phase['color']};">
                <div style="font-size:24px;">{phase['icon']}</div>
                <div style="font-weight:bold; font-size:13px;">{phase['name']}</div>
                <div style="font-size:11px; color:#888;">{phase['days']}</div>
                <div style="font-size:11px; margin-top:4px;">{phase['desc']}</div>
                <div style="margin-top:6px;">{status} {done}/{total}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ------------------------------------------------------------------
    # 实验列表
    # ------------------------------------------------------------------
    if exp_names:
        st.subheader("已记录的实验")
        for name in exp_names:
            t = ExperimentTracker.load_experiment(name, str(experiments_dir))
            s = t.summary()
            # Determine stage icon based on experiment name
            icon = "🧪"
            if "sft" in name.lower():
                icon = "🎓"
            elif "dpo" in name.lower():
                icon = "⚖️"
            elif "rl" in name.lower():
                icon = "🚀"
            with st.expander(f"{icon} 实验: {name}"):
                info_cols = st.columns(4)
                idx = 0
                for k, v in s.items():
                    if k == "experiment_name" or not isinstance(v, dict):
                        continue
                    with info_cols[idx % 4]:
                        latest = v['latest']
                        # Show delta from max for loss (lower is better)
                        if "loss" in k:
                            delta = latest - v['max']
                            st.metric(k, f"{latest:.4f}", delta=f"{delta:.4f}", delta_color="inverse")
                        else:
                            delta = latest - v['min']
                            st.metric(k, f"{latest:.4f}", delta=f"{delta:.4f}")
                    idx += 1
    else:
        st.info("暂无实验数据。请前往「训练监控」页面生成演示数据，或运行训练脚本。")


# ------------------------------------------------------------------
# 侧边栏导航
# ------------------------------------------------------------------
st.sidebar.title("LLM 训练仪表板")
st.sidebar.divider()

PAGE_OPTIONS = ["训练进度总览", "训练监控", "模型对比", "知识地图"]

page = st.sidebar.radio("导航", PAGE_OPTIONS)

# Day 快捷入口
st.sidebar.divider()
st.sidebar.markdown("#### 课程日历")
DAYS = {
    "Day 1 - PyTorch 基础": 1,
    "Day 2 - nanoGPT": 2,
    "Day 3 - HuggingFace": 3,
    "Day 4 - SFT": 4,
    "Day 5 - 数据工程": 5,
    "Day 6 - RLHF 概念": 6,
    "Day 7 - DPO": 7,
    "Day 8 - 对齐全景": 8,
    "Day 9 - RL 基础": 9,
    "Day 10 - GRPO": 10,
    "Day 11-12 - Agent RL": 11,
    "Day 13-14 - 期末项目": 13,
}

selected_day = st.sidebar.selectbox("快速跳转到天数", ["--"] + list(DAYS.keys()))
if selected_day != "--":
    st.session_state["jump_to_day"] = DAYS[selected_day]
    page = "知识地图"

# ------------------------------------------------------------------
# 页面路由
# ------------------------------------------------------------------

if page == "训练进度总览":
    _render_overview()
elif page == "训练监控":
    from dashboard.pages.training_monitor import render
    render()
elif page == "模型对比":
    from dashboard.pages.model_compare import render
    render()
elif page == "知识地图":
    from dashboard.pages.knowledge_map import render
    render()
