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


def _get_stage_color(name: str) -> str:
    """根据实验名称返回对应阶段颜色。"""
    name_lower = name.lower()
    if "sft" in name_lower:
        return "#388e3c"
    elif "dpo" in name_lower:
        return "#f57c00"
    elif "rl" in name_lower:
        return "#c62828"
    colors = ["#1976d2", "#7b1fa2", "#00838f", "#4e342e"]
    return colors[hash(name) % len(colors)]


def _get_stage_icon(name: str) -> str:
    """根据实验名称返回对应阶段图标。"""
    name_lower = name.lower()
    if "sft" in name_lower:
        return "🎓"
    elif "dpo" in name_lower:
        return "⚖️"
    elif "rl" in name_lower:
        return "🚀"
    return "🧪"


def render():
    """渲染训练监控页面。"""
    st.header("📊 训练监控")

    # 指标解读提示
    st.markdown("""
    <div style="padding:10px 16px; background:#e3f2fd; border-radius:8px; margin-bottom:16px;
                border-left:4px solid #1976d2; font-size:13px;">
        <b>📖 如何看懂训练曲线？</b><br/>
        <b>Loss（损失）</b>= 模型的"错误程度"，越低越好，训练中应该不断下降<br/>
        <b>Accuracy（准确率）</b>= 模型的"答对率"，越高越好，训练中应该不断上升<br/>
        如果 loss 不再下降甚至反弹，说明可能 <b>过拟合</b> 了
    </div>
    """, unsafe_allow_html=True)

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
        is_loss = "loss" in metric_name.lower()
        better_dir = "↓ 越低越好" if is_loss else "↑ 越高越好"

        fig = go.Figure()
        best_val = None
        best_exp = None

        for exp_name in selected:
            records = trackers[exp_name].get_metric(metric_name)
            if not records:
                continue
            steps = [r["step"] for r in records]
            values = [r["value"] for r in records]
            color = _get_stage_color(exp_name)
            icon = _get_stage_icon(exp_name)

            fig.add_trace(go.Scatter(
                x=steps, y=values, mode="lines+markers",
                name=f"{icon} {exp_name}",
                line=dict(color=color, width=2),
                marker=dict(size=4),
            ))

            # Track best final value
            final_val = values[-1] if values else None
            if final_val is not None:
                if best_val is None:
                    best_val, best_exp = final_val, exp_name
                elif is_loss and final_val < best_val:
                    best_val, best_exp = final_val, exp_name
                elif not is_loss and final_val > best_val:
                    best_val, best_exp = final_val, exp_name

        # Add annotation for best experiment
        if best_exp and len(selected) > 1:
            fig.add_annotation(
                text=f"最佳: {best_exp} ({best_val:.4f})",
                xref="paper", yref="paper",
                x=1, y=1.12,
                showarrow=False,
                font=dict(size=12, color=_get_stage_color(best_exp)),
                bgcolor="white",
                bordercolor=_get_stage_color(best_exp),
                borderwidth=1,
                borderpad=4,
            )

        fig.update_layout(
            title=dict(
                text=f"{metric_name}  <span style='font-size:12px;color:#888;'>{better_dir}</span>",
            ),
            xaxis_title="训练步数 (Step)",
            yaxis_title=metric_name,
            template="plotly_white",
            height=420,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 实验摘要表格
    # ------------------------------------------------------------------
    st.subheader("实验摘要")

    # Key insight cards
    if len(selected) > 1:
        insight_cols = st.columns(len(selected))
        for i, name in enumerate(selected):
            s = trackers[name].summary()
            icon = _get_stage_icon(name)
            color = _get_stage_color(name)
            loss_info = ""
            for m, info in s.items():
                if isinstance(info, dict) and "loss" in m:
                    improvement = ((info["max"] - info["latest"]) / info["max"] * 100) if info["max"] else 0
                    loss_info = f"Loss 下降了 {improvement:.0f}%"
                    break
            with insight_cols[i]:
                st.markdown(f"""
                <div style="text-align:center; padding:10px; border-radius:8px;
                            border-top:3px solid {color}; background:{color}11;">
                    <div style="font-size:20px;">{icon}</div>
                    <div style="font-weight:bold; font-size:14px;">{name}</div>
                    <div style="font-size:12px; color:#666; margin-top:4px;">{loss_info}</div>
                </div>
                """, unsafe_allow_html=True)

    rows = []
    for name in selected:
        s = trackers[name].summary()
        row = {"实验名称": f"{_get_stage_icon(name)} {name}"}
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
