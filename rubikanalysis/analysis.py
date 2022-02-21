"""
data analysis module.

Analysis of the preprocessed data:
1. Analyze the correlation between resource utilization, micro-indices
(independent variables) and business performance indicators QoS (dependent variables) 
2. Modeling analysis
3. Data visualization(streamlit or plotly/dash)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("# 混部干扰建模分析工具")
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
metrics.plot(x='0', y=['3','18'], subplots=True)
plt.title("nginx average response time statistics")
plt.ylabel('uint:ms')
plt.xlabel('time')
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

#  fig, ax = plt.subplots()
#  fig = metrics.plot(x='0', y=['3','18'], subplots=True)
#  st.pyplot(fig)

# fig = plt.figure()
# metrics.plot(x='0', y=['3','18'], subplots=True)
#  st.pyplot(fig)
# st.write(fig)

st.markdown("## 相关性分析")
st.markdown("### 热力图")
#  st.table(metrics)
fig, ax = plt.subplots()
sns.heatmap(metrics.corr(), ax=ax)
st.write(fig)

fig = sns.pairplot(metrics)
st.pyplot(fig)

#  g = sns.pairplot(metrics, dropna = True, hue = 'species', diag_kind="kde")
#  g.map_lower(sns.regplot)
#  st.pyplot(g)
