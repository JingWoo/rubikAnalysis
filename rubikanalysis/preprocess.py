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
import numpy as np

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

class StressProcess(object):
    def __init__(self, stress, qos, output):
        self.stress = stress
        self.qos = qos
        self.qos_index = 0
        self.output = output

    def execute(self) -> None:
        """
        Execute Stress Data Process
        """
        self.__load_and_generate_output()

    def __load_and_generate_output(self):
        # begin-timestamp end-timestamp type stress command
        stress_col_name = ['begin-timestamp', 'end-timestamp', 'type', 'stress', 'command']
        stress_table = pd.read_table(self.stress, names=stress_col_name, header=0)
        # timestamp qos
        qos_col_name = ['timestamp', 'qos']
        qos_table = pd.read_table(self.qos, names=qos_col_name, header=0)

        qos_len = len(qos_table)
        output_list = []

        no_stress_qos = self.__get_rangetime_qos(None, stress_table.at[0, 'begin-timestamp'], qos_table)
        output_list.append({"type": "none", "stress": "0", "avg-qos": no_stress_qos})

        for _, row in stress_table.iterrows():
            if self.qos_index >= qos_len:
                break

            begin_timestamp = row['begin-timestamp']
            end_timestamp = row['end-timestamp']
            average_qos = self.__get_rangetime_qos(begin_timestamp, end_timestamp, qos_table)
            output_list.append({"type": row['type'], "stress": row['stress'], "avg-qos": average_qos})

        # type stress avg-qos degradation-percent
        output_table = pd.DataFrame.from_records(output_list, columns=['type', 'stress', 'avg-qos', 'degradation-percent'])
        output_table['degradation-percent'] = 100 * (output_table['avg-qos'] - no_stress_qos) / no_stress_qos
        output_table.to_csv(self.output, index=False)
                
    def __get_rangetime_qos(self, begin_time, end_time, qos_table):
        qos_len = len(qos_table)
        if self.qos_index >= qos_len:
            return 0

        if begin_time is not None:
            while self.__compare_stimestamp_gt(begin_time, qos_table.at[self.qos_index, 'timestamp']):
                self.qos_index += 1
                if self.qos_index >= qos_len:
                    return 0
        begin_index = self.qos_index

        while self.__compare_stimestamp_gt(end_time, qos_table.at[self.qos_index, 'timestamp']):
            self.qos_index += 1
            if self.qos_index >= qos_len:
                break
        end_index = self.qos_index

        return np.mean(qos_table[begin_index:end_index]["qos"])

    def __compare_stimestamp_gt(self, time1, time2):
        time1_st = int(time.mktime(time.strptime(
            time1, "%Y-%m-%d %H:%M:%S")))
        time2_st = int(time.mktime(time.strptime(
            time2, "%Y-%m-%d %H:%M:%S")))

        return time1_st > time2_st
