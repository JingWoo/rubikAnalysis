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

from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from enum import unique
from itertools import product
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


def get_stress_chart(data, symbol):
    hover = alt.selection_single(
        fields=["stress"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title="Qos variation with stress " + symbol)
        .mark_line()
        .encode(
            x=alt.X("stress:Q", axis=alt.Axis(orient="top")),
            y=alt.Y("degradation-percent:Q", sort='descending', title="degradation percent(%)"),
            color="type",
            strokeDash="type",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="stress:Q",
            y="degradation-percent:Q",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("stress", title="Stress Info"),
                alt.Tooltip("avg-qos", title="Qos"),
                alt.Tooltip("degradation-percent", title="Degradation Percent")
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()


def stress_sensitivity(stress_degrade):
    if stress_degrade['degradation-percent'] <= 5:
        return "no"

    if stress_degrade['degradation-percent'] <= 10:
        return "low"

    if stress_degrade['degradation-percent'] <= 20:
        return "medium"

    return "high"


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
st.markdown("## 业务介绍 TODO")
st.markdown("## 环境说明")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col="Item")
    st.write(data)

node_info = pd.read_csv("../tests/data/node.csv", index_col="Item")
st.table(node_info)

st.markdown("## 指标数据")
data = pd.read_csv("../tests/data/merge.csv")
mode = st.radio(
    "Please select a mode to visualize the data:",
    ('origin', 'normalization', 'standardization'))

if mode == 'normalization':
    normalize_table(data)
elif mode == 'standardization':
    standardize_table(data)

all_symbols = list(data.columns[1:])
symbols = st.multiselect("Choose metrics to visualize",
                         all_symbols, all_symbols[:3])
symbols.insert(0, data.columns[0])

source = data[symbols]

chart = get_chart(source)
st.altair_chart(chart, use_container_width=True)

st.markdown("## 资源敏感度分析")

# type stress avg-qos degradation-percent
stress = pd.read_csv("../tests/data/stress.csv", keep_default_na=False)
stress_all_symbols = ["cpu", "memory", "disk io", "cache", "network"]
stress_unique_symbols = stress.type.unique()

stress_symbols = []
for usymbol, asymbol in product(stress_unique_symbols, stress_all_symbols):
    if asymbol in usymbol and asymbol not in stress_symbols:
        stress_symbols.append(asymbol)
stress_symbols = st.multiselect("Choose metrics to visualize",
                                stress_symbols, stress_symbols)

for stress_symbol in stress_symbols:
    stress_source = stress[stress.type.str.contains(stress_symbol)]

    # 插入无压力数据
    nstress = stress[stress.type == "none"]
    for usymbol in stress_unique_symbols:
        if stress_symbol in usymbol:
            stress_source = pd.concat([stress_source, nstress.replace(
                "none", usymbol)], axis=0, ignore_index=True)

    stress_chart = get_stress_chart(stress_source, stress_symbol)
    st.altair_chart(stress_chart, use_container_width=True)

st.markdown("#### 资源敏感度排序")

stress_degrade = (
    stress.drop(stress[stress.type == "none"].index)[
        ['type', 'degradation-percent']]
    .groupby(by='type')
    .max()
    .sort_values(by='degradation-percent', ascending=False)
)

stress_degrade.loc[:, 'sensitivity'] = stress_degrade.apply(
    stress_sensitivity, axis=1)
st.table(stress_degrade)

st.info("degradation-percent in (, 5]:no ; (5, 10]:low ; (10, 20]:medinum ; (20,):high")

st.markdown("## 相关性分析")
st.markdown("### 热力图")
# pearson相关系数:  连续、正态分布、线性数据
# spearman相关系数: 非线性的、非正态数据
# Kendall相关系数:  分类变量、无序数据
fig, ax = plt.subplots()
metrics_correlation = data.corr(method="pearson")
sns.heatmap(metrics_correlation, ax=ax)
st.write(fig)

# fig = sns.pairplot(data)
# st.pyplot(fig)

fig = sns.jointplot(x='branch_misses', y='qos', data=data, kind='reg')
st.pyplot(fig)

fig = sns.jointplot(x='topdown_bad_spec', y='qos', data=data, kind='reg')
st.pyplot(fig)

st.info("|r|>0.95存在显著性相关;|r|≥0.8高度相关;0.5≤|r|<0.8 中度相关;0.3≤|r|<0.5低度相关;|r|<0.3关系极弱")
st.markdown("### 相关性指标排序")
metrics_correlation.iloc[-1] = abs(metrics_correlation.iloc[-1])
st.table(metrics_correlation.iloc[-1].sort_values(ascending=False))


vaild_metrics = metrics_correlation[metrics_correlation["qos"] > 0.2].index.tolist(
)
vaild_metrics.remove("qos")
st.markdown("#### 筛选有效指标")
col = st.columns(len(vaild_metrics))
for i in range(len(vaild_metrics)):
    corr_value = "{:.5}".format(metrics_correlation.iloc[-1][vaild_metrics[i]])
    col[i].metric(vaild_metrics[i], corr_value, i + 1)

st.markdown("### 回归拟合分析")
# 数据预处理: 去除无效值; 特性缩放:标准化; 模型训练
x = data[vaild_metrics]
y = data[["qos"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)


def draw_comparison_altair_chart(y_test, y_pred):
    y_test_list = y_test["qos"]
    y_pred_list = y_pred if type(y_pred[0]) is not np.ndarray else [
        i[0] for i in y_pred]
    list_of_tuples = list(zip(y_test_list, y_pred_list))
    source = pd.DataFrame(list_of_tuples, columns=['Measured', 'Predicted'],
                          index=pd.RangeIndex(len(y_pred), name='index'))
    source = source.reset_index().melt('index', var_name='category', value_name='qos')
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['index'], empty='none')
    line = alt.Chart(source).mark_line(interpolate='basis').encode(
        x='index:Q',
        y='qos:Q',
        color='category:N'
    )

    selectors = alt.Chart(source).mark_point().encode(
        x='index:Q',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'qos:Q', alt.value(' '))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x='index:Q',
    ).transform_filter(
        nearest
    )

    charts = alt.layer(
        line, selectors, points, rules, text
    ).interactive()
    st.altair_chart(charts, use_container_width=True)


def draw_comparison_matplotlib_chart(y_test, y_pred):
    fig = plt.figure()
    plt.plot(np.arange(len(y_pred)),
             y_test[["qos"]].values, 'go-', label='Measured')
    plt.plot(np.arange(len(y_pred)), y_pred, 'ro-', label='Predicted')
    plt.title("Interference Model Analysis")
    plt.xlabel("Index")
    plt.ylabel("QoS")
    plt.legend()
    st.pyplot(fig)


def train_and_test_model(model):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # score = model.score(x_test, y_test)
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    draw_comparison_altair_chart(y_test, y_pred)

    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    st.markdown("#### 性能度量")
    model_evaluation = "MSE: {}, EMSE: {}".format(mse, rmse)
    st.write(model_evaluation)


analysis_model = st.selectbox(
    'Please select a regression model',
    ('Decision Tree Regression', 'Linear Regression', 'SVM Regression',
     "KNN Regression", "Random Forest Regression", "Adaboost Regression",
     "Gradient Boosting Regression", "Bagging Regression", "ExtraTree Regression",))

if analysis_model == 'Decision Tree Regression':
    from sklearn import tree
    regressor = tree.DecisionTreeRegressor()
    train_and_test_model(regressor)
elif analysis_model == 'Linear Regression':
    regressor = linear_model.LinearRegression()
    train_and_test_model(regressor)
elif analysis_model == 'SVM Regression':
    from sklearn import svm
    regressor = svm.SVR()
    train_and_test_model(regressor)
elif analysis_model == 'KNN Regression':
    from sklearn import neighbors
    regressor = neighbors.KNeighborsRegressor()
    train_and_test_model(regressor)
elif analysis_model == 'Random Forest Regression':
    from sklearn import ensemble
    regressor = ensemble.RandomForestRegressor(n_estimators=20)
    train_and_test_model(regressor)
elif analysis_model == 'Adaboost Regression':
    from sklearn import ensemble
    regressor = ensemble.AdaBoostRegressor(n_estimators=50)
    train_and_test_model(regressor)
elif analysis_model == 'Gradient Boosting Regression':
    from sklearn import ensemble
    regressor = ensemble.GradientBoostingRegressor(n_estimators=100)
    train_and_test_model(regressor)
elif analysis_model == 'Bagging Regression':
    from sklearn import ensemble
    regressor = ensemble.BaggingRegressor()
    train_and_test_model(regressor)
elif analysis_model == 'ExtraTree Regression':
    from sklearn import tree
    regressor = tree.ExtraTreeRegressor()
    train_and_test_model(regressor)
