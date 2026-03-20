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

    # 状态统计
    col1, col2, col3 = st.columns(3)
    col1.metric("实验总数", len(exp_names))

    # 课程目录状态
    day_folders = [
        "day01_pytorch_basics", "day02_nanoGPT", "day03_huggingface",
        "day04_sft", "day05_data_engineering", "day06_rlhf_concepts",
        "day07_dpo", "day08_alignment_landscape", "day09_rl_basics",
        "day10_grpo", "day11_12_agent_rl", "day13_14_final_project",
    ]
    existing = sum(1 for f in day_folders if (PROJECT_ROOT / f).exists())
    col2.metric("已创建课程目录", f"{existing}/{len(day_folders)}")
    col3.metric("训练阶段", "Base -> SFT -> DPO -> RL")

    st.divider()

    # 实验列表
    if exp_names:
        st.subheader("已记录的实验")
        for name in exp_names:
            t = ExperimentTracker.load_experiment(name, str(experiments_dir))
            s = t.summary()
            with st.expander(f"实验: {name}"):
                info_cols = st.columns(4)
                idx = 0
                for k, v in s.items():
                    if k == "experiment_name" or not isinstance(v, dict):
                        continue
                    with info_cols[idx % 4]:
                        st.metric(k, f"{v['latest']:.4f}")
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
