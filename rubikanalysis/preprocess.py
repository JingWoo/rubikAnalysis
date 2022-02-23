"""
Indicator data preprocessing module.

The low-level metrics do not correspond to the high-level SLAs,
and the indicators are corresponding and merged according to
the method of time stamp proximity search..
"""

from pydoc import doc
import time
import sys
import pandas as pd


class Preprocess(object):
    def __init__(self, metrics, qos, output):
        self.metrics = metrics
        self.qos = qos
        self.output = output

    def execute(self) -> None:
        """
        Execute preprocessing
        """
        self.__load_and_generate_output()

    def __load_and_generate_output(self):
        # timestamp ipc cache-misses context-switch
        # metrics = pd.read_table(self.metrics, sep="\t", names=[
        #                         "timestamp", "ipc", "cache-misses"])
        # print(metrics["timestamp"])
        metrics = pd.read_table(self.metrics, header=None, index_col=0)
        qos = pd.read_table(self.qos, header=None, index_col=0)
        metrics_timestamps = list(metrics.index)
        qos_timestamps = list(qos.index)
        metrics_filted_result = []
        qos_filted_result = []
        if len(metrics_timestamps) > len(qos_timestamps):
            metrics_filted_result, qos_filted_result = self.__match_and_filter(
                metrics_timestamps, qos_timestamps)
        else:
            qos_filted_result, metrics_filted_result = self.__match_and_filter(
                qos_timestamps, metrics_timestamps)

        output_table = metrics.loc[metrics_filted_result]
        qos_table = qos.loc[qos_filted_result]
        qos_table.to_csv("qos.csv")
        col = qos_table.iloc[:, 0]
        output_table[output_table.shape[1]+1] = col.values
        output_table.to_csv(self.output)

    def __match_and_filter(self, broad_pd, narrow_pd):
        i = 0
        diff = sys.maxsize
        broad_tss = []
        narrow_tss = []
        for index_ts in narrow_pd:
            base_ts = int(time.mktime(
                time.strptime(index_ts, "%Y-%m-%d %H:%M:%S")))
            cmp_ts = int(time.mktime(time.strptime(
                broad_pd[i], "%Y-%m-%d %H:%M:%S")))
            while abs(cmp_ts-base_ts) < diff and i < len(broad_pd) - 1:
                diff = min(diff, abs(cmp_ts-base_ts))
                i = i + 1
                cmp_ts = int(time.mktime(time.strptime(
                    broad_pd[i], "%Y-%m-%d %H:%M:%S")))
            broad_tss.append(broad_pd[i-1])
            narrow_tss.append(index_ts)
            diff = sys.maxsize

        return broad_tss, narrow_tss
