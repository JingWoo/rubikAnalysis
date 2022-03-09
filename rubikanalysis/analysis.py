"""
data analysis module.

Analysis of the preprocessed data:
1. Analyze the correlation between resource utilization, micro-indices
(independent variables) and business performance indicators QoS (dependent variables)
2. Modeling analysis
3. Data visualization(streamlit or plotly/dash)
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
            y=alt.Y("degradation-percent:Q", sort='descending',
                    title="degradation percent(%)"),
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

node_info = pd.read_csv("../tests/data/default/node.csv", index_col="Item")
uploaded_node_file = st.file_uploader("上传测试环境配置", type="csv")
if uploaded_node_file is not None:
    node_info = pd.read_csv(uploaded_node_file, index_col="Item")

st.table(node_info)

st.markdown("## 指标数据")
# data = pd.read_csv("../tests/data/default/nginx.csv")
data = pd.read_csv("../tests/data/clickhouse/l3cache_stress.csv")
uploaded_metrics_file = st.file_uploader("上传指标数据-QoS数据", type="csv")
if uploaded_metrics_file is not None:
    data = pd.read_csv(uploaded_metrics_file)

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
stress = pd.read_csv("../tests/data/default/stress.csv", keep_default_na=False)
uploaded_stress_file = st.file_uploader("上传压力测试指标数据", type="csv")
if uploaded_stress_file is not None:
    stress = pd.read_csv(uploaded_stress_file)

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

st.info(
    "degradation-percent in (, 5]:no ; (5, 10]:low ; (10, 20]:medinum ; (20,):high")

st.markdown("## 相关性分析")
st.markdown("### 热力图")
# pearson相关系数:  连续、正态分布、线性数据
# spearman相关系数: 非线性的、非正态数据
# Kendall相关系数:  分类变量、无序数据
fig, ax = plt.subplots()
corr_mode = st.radio(
    """选择相关性分析方法:
    pearson: 连续、正态分布、线性数据;
    spearman: 非线性的、非正态数据;
    kendall: 分类变量、无序数据""",
    ('pearson', 'spearman', 'Kendall'))

if corr_mode == 'spearman':
    metrics_correlation = data.corr(method="spearman")
elif corr_mode == 'Kendall':
    metrics_correlation = data.corr(method="kendall")
else:
    metrics_correlation = data.corr(method="pearson")

sns.heatmap(metrics_correlation, ax=ax)
st.write(fig)

# fig = sns.pairplot(data)
# st.pyplot(fig)

st.info("|r|>0.95存在显著性相关;|r|≥0.8高度相关;0.5≤|r|<0.8 中度相关;0.3≤|r|<0.5低度相关;|r|<0.3关系极弱")
st.markdown("### 相关性指标排序")
sorted_metrics_correlation = abs(metrics_correlation.iloc[-1]).sort_values(
    ascending=False)
st.table(sorted_metrics_correlation)

vaild_metrics = sorted_metrics_correlation[abs(
    metrics_correlation["qos"]) > 0.3].index.tolist()
vaild_metrics.remove("qos")

st.markdown("#### 筛选有效指标")
col = st.columns(len(vaild_metrics))
label_col = 5
label_raw = len(vaild_metrics) // label_col + 1

for raw in range(label_raw):
    show_label_count = label_col if (len(
        vaild_metrics) - raw * label_col) // label_col > 0 else len(vaild_metrics) % label_col
    stcolumns = st.columns(show_label_count)
    for col in range(show_label_count):
        index = raw * label_col + col
        corr_value = "{:.5}".format(
            metrics_correlation.iloc[-1][vaild_metrics[index]])
        stcolumns[col].metric(vaild_metrics[index], corr_value, index + 1)

for i in range(len(vaild_metrics)):
    if i >= 5:
        break
    fig = sns.jointplot(x=vaild_metrics[i], y='qos', data=data, kind='reg')
    st.pyplot(fig)


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
