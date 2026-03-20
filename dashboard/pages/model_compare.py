"""
模型对比页面 - 加载不同阶段的模型，并排显示输出，支持打分和标注。
"""

import json
import sys
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 阶段配色和元数据
STAGE_META = {
    "Base": {"icon": "📚", "color": "#1976d2", "bg": "#e3f2fd", "desc": "未经训练的基座模型"},
    "SFT": {"icon": "🎓", "color": "#388e3c", "bg": "#e8f5e9", "desc": "监督微调后，学会了格式"},
    "DPO": {"icon": "⚖️", "color": "#f57c00", "bg": "#fff3e0", "desc": "偏好对齐后，分清好坏"},
    "RL": {"icon": "🚀", "color": "#c62828", "bg": "#fce4ec", "desc": "强化学习后，能力更强"},
}

# 偏好数据保存目录
PREFERENCE_DIR = PROJECT_ROOT / "experiments" / "preferences"


def render():
    """渲染模型对比页面。"""
    st.header("🔄 模型对比")

    # 直观解释
    st.markdown("""
    <div style="padding:10px 16px; background:#e3f2fd; border-radius:8px; margin-bottom:16px;
                border-left:4px solid #1976d2; font-size:13px;">
        <b>📖 模型对比是在看什么？</b><br/>
        同一个问题，分别让 <b>不同训练阶段</b> 的模型来回答，直观对比效果差异。<br/>
        就像同一道考题让 "新手→学徒→老手→专家" 分别作答，看看谁答得更好。
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 模式选择：真实模型 vs 模拟模式
    # ------------------------------------------------------------------
    mode = st.radio("运行模式", ["模拟模式（无需 GPU）", "真实模型加载"], horizontal=True)

    if mode == "真实模型加载":
        _render_real_mode()
    else:
        _render_mock_mode()


def _render_mock_mode():
    """模拟模式：无需真实模型，快速体验界面功能。"""
    st.caption("当前为模拟模式，输出为预设示例文本。切换到"真实模型加载"以使用真实模型。")

    stages = ["Base", "SFT", "DPO", "RL"]
    selected_stages = st.multiselect("选择对比阶段", stages, default=stages)

    prompt = st.text_area("输入 Prompt", value="请解释什么是强化学习中的 PPO 算法。", height=100)

    if st.button("生成对比", type="primary") and prompt.strip():
        mock_outputs = {
            "Base": f"PPO 是一个算法。它用于训练。{prompt[:10]}...",
            "SFT": "PPO（Proximal Policy Optimization）是一种策略梯度强化学习算法，"
                   "由 OpenAI 于 2017 年提出。它通过限制每次策略更新的幅度来保证训练稳定性，"
                   "使用 clip 机制将比率限制在 [1-ε, 1+ε] 范围内。",
            "DPO": "PPO（近端策略优化）是强化学习中最流行的算法之一。它的核心思想是：\n\n"
                   "1. 使用重要性采样来复用旧策略的数据\n"
                   "2. 通过裁剪目标函数来防止策略更新过大\n"
                   "3. 同时优化策略网络和价值网络\n\n"
                   "PPO 在大语言模型对齐中被广泛使用，是 RLHF 的关键组件。",
            "RL": "PPO（Proximal Policy Optimization）是一种 on-policy 的策略梯度方法。\n\n"
                  "**核心机制：**\n"
                  "- Clipped Surrogate Objective：裁剪概率比率，限制更新步长\n"
                  "- GAE（Generalized Advantage Estimation）：平衡偏差与方差\n"
                  "- 多 epoch 小批量更新：提高数据利用效率\n\n"
                  "**在 LLM 对齐中的应用：**\n"
                  "在 RLHF 流程中，PPO 用于根据人类偏好的奖励模型来微调语言模型，"
                  "使其生成更符合人类期望的回答。",
        }

        _display_outputs(prompt, selected_stages, mock_outputs)


def _render_real_mode():
    """真实模型加载模式。"""
    st.caption("请在侧边栏输入模型路径并加载。")

    # 在 session_state 中维护 comparator 实例
    if "comparator" not in st.session_state:
        from utils.model_comparator import ModelComparator
        st.session_state.comparator = ModelComparator()

    comparator = st.session_state.comparator

    # 模型加载区
    with st.expander("模型管理", expanded=not comparator.list_models()):
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.text_input("模型名称", placeholder="例如：sft")
        with col2:
            model_path = st.text_input("模型路径", placeholder="例如：Qwen/Qwen2-0.5B")

        if st.button("加载模型") and model_name and model_path:
            with st.spinner(f"正在加载 {model_name}..."):
                try:
                    comparator.load_model(model_name, model_path)
                    st.success(f"模型 '{model_name}' 加载成功！")
                except Exception as e:
                    st.error(f"加载失败：{e}")

        loaded = comparator.list_models()
        if loaded:
            st.write(f"已加载模型：{', '.join(loaded)}")

    if not comparator.list_models():
        st.info("请先加载至少一个模型。")
        return

    # 生成区
    prompt = st.text_area("输入 Prompt", height=100)
    col_t, col_p = st.columns(2)
    temperature = col_t.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = col_p.slider("Max New Tokens", 32, 1024, 256, 32)

    if st.button("生成对比", type="primary") and prompt.strip():
        outputs = {}
        for name in comparator.list_models():
            with st.spinner(f"正在用 {name} 生成..."):
                result = comparator.generate(name, prompt, max_new_tokens=max_tokens, temperature=temperature)
                outputs[name] = result.output

        _display_outputs(prompt, comparator.list_models(), outputs)


def _display_outputs(prompt: str, stages, outputs: dict):
    """显示各模型输出，并提供打分和标注功能。"""
    st.subheader("对比结果")

    cols = st.columns(len(stages))
    scores = {}

    for i, stage in enumerate(stages):
        meta = STAGE_META.get(stage, {"icon": "🧪", "color": "#666", "bg": "#f5f5f5", "desc": ""})
        with cols[i]:
            # Stage header with icon and color
            st.markdown(f"""
            <div style="text-align:center; padding:8px; border-radius:8px;
                        background:{meta['bg']}; border-top:3px solid {meta['color']}; margin-bottom:8px;">
                <div style="font-size:24px;">{meta['icon']}</div>
                <div style="font-weight:bold; color:{meta['color']};">{stage}</div>
                <div style="font-size:11px; color:#888;">{meta['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

            output_text = outputs.get(stage, "（未生成）")
            st.text_area(f"输出 - {stage}", value=output_text, height=250, disabled=True, key=f"out_{stage}")

            # Star rating visualization
            score = st.slider(f"评分", 1, 5, 3, key=f"score_{stage}")
            scores[stage] = score
            stars = "★" * score + "☆" * (5 - score)
            st.markdown(f"<div style='text-align:center; font-size:18px; color:#f57c00;'>{stars}</div>",
                        unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 评分雷达图对比
    # ------------------------------------------------------------------
    if len(stages) >= 2 and all(s in scores for s in stages):
        st.subheader("评分对比")
        dimensions = ["回答质量", "格式规范", "内容完整", "表达清晰", "综合评分"]

        fig = go.Figure()
        for stage in stages:
            meta = STAGE_META.get(stage, {"color": "#666"})
            # Use the score to generate simulated dimension scores for demonstration
            base_score = scores[stage]
            dim_values = [
                max(1, min(5, base_score + (hash(f"{stage}{d}") % 3 - 1) * 0.3))
                for d in dimensions
            ]
            dim_values.append(dim_values[0])  # Close the polygon
            fig.add_trace(go.Scatterpolar(
                r=dim_values,
                theta=dimensions + [dimensions[0]],
                fill="toself",
                name=f"{STAGE_META.get(stage, {}).get('icon', '')} {stage}",
                line=dict(color=meta["color"]),
                opacity=0.6,
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5.5]),
            ),
            showlegend=True,
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 偏好标注
    # ------------------------------------------------------------------
    st.subheader("偏好标注")
    st.markdown("""
    <div style="padding:8px 12px; background:#fff3e0; border-radius:6px; font-size:12px; margin-bottom:12px;">
        <b>什么是偏好标注？</b> 选出你认为更好和更差的回答，这类数据正是 DPO 训练所需要的！
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        chosen = st.selectbox("选择更好的输出（Chosen）", stages)
    with col_b:
        rejected = st.selectbox("选择较差的输出（Rejected）", [s for s in stages if s != chosen])

    notes = st.text_input("备注（可选）")

    if st.button("保存偏好数据"):
        PREFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "prompt": prompt,
            "chosen": {"model": chosen, "output": outputs.get(chosen, "")},
            "rejected": {"model": rejected, "output": outputs.get(rejected, "")},
            "scores": scores,
            "notes": notes,
            "timestamp": time.time(),
        }
        filename = PREFERENCE_DIR / f"pref_{int(time.time())}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        st.success(f"偏好数据已保存到 {filename.name}")
