"""
data analysis module.

Analysis of the preprocessed data:
1. Analyze the correlation between resource utilization, micro-indices
(independent variables) and business performance indicators QoS (dependent variables)
2. Modeling analysis
3. Data visualization(streamlit or plotly/dash)
混部建模分析工具通过在实际节点上运行，通过软硬协同分析的方式，对实时收集的节点内Pod资源使用量和
RDT、eBPF、perf等硬件、内核及QoS业务等相关指标进行分析，研究负载执行时的应用级特征和系统层特征，
分析业务对资源的敏感度，识别节点内在线业务QoS是否符合预期以及业务之间性能干扰导致的性能下降的因素，
推算并反馈更优的部署方式，知道云集群的资源规划和调度管理，减少在离线业务竞争共享资源（如CPU、缓存、
内存带宽等）导致的业务性能下降，优化资源配比，实现资源均衡错峰互补使用，最大化提升节点资源利用，保障
在线业务QoS不受影响

在混部集群中，性能干扰会严重影响在线业务的实时性和稳定性，同时也会降低离线业务的吞吐率。因此，混部集群
管理系统必须有效控制性能干扰，利用资源隔离技术根据性能干扰变化快速、及时底动态调整运行在离线业务的资源
供给。然而，由于业务负载的动态性、资源需求的多样性等导致性能干扰复杂性剧增，呈现出性能对干扰的敏感性具
有动态变化、模式多样复杂等特点。混部建模工具旨在通过对软、英指标的分析解决在离线混部业务的性能干扰问题，
以预测作业在动态负载、资源竞争及干扰模式等条件下的性能，快速识别当前业务对各维度资源的敏感性，指导集群
资源规划、调度策略优化及节点在离线资源合理配比，同时探索理论可行的高精度、地开销方法，以接近最优解的方式
将复杂问题简单化，检测业务是否收到干扰及干扰来源

混部建模分析采用分阶段、循序渐进的方式完善。第一阶段基于历史监控数据的数据获取方法，通过外部压力注入获取
业务运行产生的监控数据（负载当前资源使用、业务运行对应的硬件、内核等底层深度指标（自变量）及业务性能（因变量））
中提取性能干扰模型所需要的数据集，分析业务对不同维度资源的敏感度，实现对业务类型分类标定，同时根据自变量指标
与因变量指标的相关性分析、显著性验证等手段，分析并筛选出有效指标对百合混部业务场景的调度及资源管理进行指导；
在第二阶段中，根据第一阶段在不同配置、不同混部模式下负载执行时的应用级和系统级特征，深入系统底层通用性能指标
与上层应用性能建模，对应用干扰性能探索提炼，指导黑盒在离线混部场景资源规划、调度管理
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
        alt.Chart(data.reset_index(drop=True).melt("timestamp"),
                  title="Metrics variation")
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
        alt.Chart(data.reset_index(drop=True).melt("timestamp"),
                  title="Metrics variation")
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

    return (lines + points + tooltips).interactive()


def normalize_table(df):
    cols = list(df)
    for item in cols:
        if df[item].dtype == 'int64' or df[item].dtype == 'float64':
            max_tmp = np.max(np.array(df[item]))
            min_tmp = np.min(np.array(df[item]))
            if (max_tmp != min_tmp):
                df[item] = df[item].apply(
                    lambda x: (x - min_tmp) * 100 / (max_tmp - min_tmp))


def standardize_table(df):
    cols = list(df)
    for item in cols:
        if df[item].dtype == 'int64' or df[item].dtype == 'float64':
            mean_tmp = np.mean(np.array(df[item]))
            std_tmp = np.std(np.array(df[item]))
            if(std_tmp):
                df[item] = df[item].apply(lambda x: (x - mean_tmp) / std_tmp)


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
metrics = pd.read_csv("merge.csv")
mode = st.radio(
    "Please select a mode to visualize the data:",
    ('origin', 'normalization', 'standardization'))

if mode == 'normalization':
    normalize_table(metrics)
elif mode == 'standardization':
    standardize_table(metrics)

all_symbols = list(metrics.columns[1:])
symbols = st.multiselect("Choose metrics to visualize",
                         all_symbols, all_symbols[:3])
symbols.insert(0, metrics.columns[0])

source = metrics[symbols]

chart = get_chart(source)
st.altair_chart(chart, use_container_width=True)

st.markdown("## 资源敏感度分析")

col1, col2, col3 = st.columns(3)
col1.metric("IPC", "0.525", "1")
col2.metric("Banch Misses", "0.328", "2")
col3.metric("Cache Misses", "0.234", "3")
st.markdown("## 相关性分析")
st.markdown("### 热力图")
# pearson相关系数:  连续、正态分布、线性数据
# spearman相关系数: 非线性的、非正态数据
# Kendall相关系数:  分类变量、无序数据
fig, ax = plt.subplots()
metrics_correlation = metrics.corr(method="pearson")

sns.heatmap(metrics_correlation, ax=ax)
st.write(fig)
st.info("|r|>0.95存在显著性相关;|r|≥0.8高度相关;0.5≤|r|<0.8 中度相关;0.3≤|r|<0.5低度相关;|r|<0.3关系极弱")
metrics_correlation.iloc[-1] = abs(metrics_correlation.iloc[-1])
st.table(metrics_correlation.iloc[-1].sort_values(ascending=False))

# fig = sns.pairplot(metrics)
# st.pyplot(fig)
st.markdown("### 相关性指标排序")

st.markdown("### 回归拟合分析")
