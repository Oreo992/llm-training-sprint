"""
训练监控页面 - 实时显示 loss/accuracy 曲线，支持多实验叠加对比。
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_tracker import ExperimentTracker


def render():
    """渲染训练监控页面。"""
    st.header("📊 训练监控")

    experiments_dir = PROJECT_ROOT / "experiments"
    exp_names = ExperimentTracker.list_experiments(str(experiments_dir))

    if not exp_names:
        st.info("暂无实验数据。请先运行训练脚本或使用下方按钮生成演示数据。")
        if st.button("生成演示数据"):
            _create_demo_data(experiments_dir)
            st.rerun()
        return

    # 选择要对比的实验
    selected = st.multiselect("选择实验进行对比", exp_names, default=exp_names[:2])

    if not selected:
        st.warning("请至少选择一个实验。")
        return

    # 选择要查看的指标
    all_metrics = set()
    trackers = {}
    for name in selected:
        t = ExperimentTracker.load_experiment(name, str(experiments_dir))
        trackers[name] = t
        all_metrics.update(t.metrics.keys())

    all_metrics = sorted(all_metrics)
    if not all_metrics:
        st.warning("所选实验中没有记录任何指标。")
        return

    selected_metrics = st.multiselect("选择指标", all_metrics, default=all_metrics[:2])

    # 为每个指标绘制曲线
    for metric_name in selected_metrics:
        fig = go.Figure()
        for exp_name in selected:
            records = trackers[exp_name].get_metric(metric_name)
            if not records:
                continue
            steps = [r["step"] for r in records]
            values = [r["value"] for r in records]
            fig.add_trace(go.Scatter(x=steps, y=values, mode="lines+markers", name=exp_name))

        fig.update_layout(
            title=metric_name,
            xaxis_title="Step",
            yaxis_title=metric_name,
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # 实验摘要表格
    st.subheader("实验摘要")
    rows = []
    for name in selected:
        s = trackers[name].summary()
        row = {"实验名称": name}
        for m, info in s.items():
            if m == "experiment_name":
                continue
            if isinstance(info, dict):
                row[f"{m} (最新)"] = f"{info['latest']:.4f}"
                row[f"{m} (最小)"] = f"{info['min']:.4f}"
                row[f"{m} (最大)"] = f"{info['max']:.4f}"
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _create_demo_data(experiments_dir: Path):
    """创建演示实验数据。"""
    import math
    import random

    for exp_name, base_loss, acc_start in [
        ("sft_exp_01", 2.5, 0.3),
        ("dpo_exp_01", 2.0, 0.4),
        ("rl_exp_01", 1.8, 0.45),
    ]:
        t = ExperimentTracker(exp_name, str(experiments_dir), tags={"type": exp_name.split("_")[0]})
        for step in range(100):
            loss = base_loss * math.exp(-0.02 * step) + random.gauss(0, 0.05)
            acc = min(acc_start + 0.005 * step + random.gauss(0, 0.01), 0.98)
            t.log_metric("loss", loss, step)
            t.log_metric("accuracy", acc, step)
        t.log_text("sample_output", f"这是 {exp_name} 的示例模型输出。")

    st.success("已生成 3 组演示实验数据。")
