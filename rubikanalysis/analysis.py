"""
data analysis module.

Analysis of the preprocessed data:
1. Analyze the correlation between resource utilization, micro-indices
(independent variables) and business performance indicators QoS (dependent variables)
2. Modeling analysis
3. Data visualization(streamlit or plotly/dash)
"""

from enum import unique
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt


def get_chart(data):
    hover = alt.selection_single(
        fields=["timestamp"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title="Evolution of stock prices")
        .mark_line()
        .encode(
            x="timestamp:T",
            y="value:Q",
            color="variable",
            strokeDash="symbol",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="timestamp",
            y="value",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("timestamp", title="Date"),
                alt.Tooltip("value", title="Price (USD)"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()


st.set_page_config(layout="centered", page_icon="⎈",
                   page_title="rubik analysis")

st.markdown("# ⎈ 混部干扰建模分析工具")
st.markdown("## 业务介绍")
st.markdown("## 环境说明")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col="Item")
    st.write(data)

node_info = pd.read_csv("node.csv", index_col="Item")
st.table(node_info)

st.markdown("## 指标数据")

# metrics = pd.read_csv("merge.csv", index_col="timestamp")
metrics = pd.read_csv("merge.csv")
all_symbols = list(metrics.columns[1:])
symbols = st.multiselect("Choose metrics to visualize",
                         all_symbols, all_symbols[:3])
symbols.insert(0, metrics.columns[0])

source = metrics[symbols]

hover = alt.selection_single(
    fields=["timestamp"],
    nearest=True,
    on="mouseover",
    empty="none",
)

lines = (
    alt.Chart(source.reset_index().melt("timestamp"),
              title="Evolution of stock prices")
    .mark_line()
    .encode(
        x="timestamp:T",
        y="value:Q",
        color="variable",
        # strokeDash="variable",
    ).properties(width=800, height=400)
)

# Draw points on the line, and highlight based on selection
points = lines.transform_filter(hover).mark_circle(size=65)

# Draw a rule at the location of the selection
tooltips = (
    alt.Chart(source)
    .mark_rule()
    .encode(
        x="timestamp:T",
        y="value:Q",
        opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
        tooltip=[
            alt.Tooltip("timestamp", title="Time"),
            alt.Tooltip("value", title="Metrics"),
        ],
    )
    .add_selection(hover)
)

# chart = (lines + points + tooltips).interactive()
chart = (lines).interactive()


st.altair_chart(chart, use_container_width=True)
# source = metrics[["perf_ipc", "branch_misses"]]
# chart = get_chart(source)
#st.altair_chart(chart, use_container_width=True)
# st.line_chart(metrics[["perf_ipc", "branch_misses"]])


c = alt.Chart(metrics).mark_line().encode(
    x='timestamp:T',
    y='perf_ipc:Q'
)
st.altair_chart(c, use_container_width=True)

c = alt.Chart(metrics).mark_line().encode(
    x='timestamp:T',
    y='qos:Q'
)
st.altair_chart(c, use_container_width=True)

# metrics.plot(x='timestamp', y=['perf_ipc', 'qos'], subplots=True)
# plt.title("nginx average response time statistics")
# plt.ylabel('uint:ms')
# plt.xlabel('time')
# plt.show()
# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.pyplot()

#  fig, ax = plt.subplots()
#  fig = metrics.plot(x='0', y=['3','18'], subplots=True)
#  st.pyplot(fig)

# fig = plt.figure()
# metrics.plot(x='0', y=['3','18'], subplots=True)
#  st.pyplot(fig)
# st.write(fig)
st.markdown("## 资源敏感度分析")

col1, col2, col3 = st.columns(3)
col1.metric("IPC", "0.525", "1")
col2.metric("Banch Misses", "0.328", "2")
col3.metric("Cache Misses", "0.234", "3")
st.markdown("## 相关性分析")
st.markdown("### 热力图")
fig, ax = plt.subplots()
sns.heatmap(metrics.corr(), ax=ax)
st.write(fig)
st.info("|r|>0.95存在显著性相关;|r|≥0.8高度相关;0.5≤|r|<0.8 中度相关;0.3≤|r|<0.5低度相关;|r|<0.3关系极弱")
# fig = sns.pairplot(metrics)
# st.pyplot(fig)
st.markdown("### 相关性指标排序")

#  g = sns.pairplot(metrics, dropna = True, hue = 'species', diag_kind="kde")
#  g.map_lower(sns.regplot)
#  st.pyplot(g)
st.markdown("### 回归拟合分析")
