'''
@Author: your name
@Date: 2020-07-18 17:21:39
@LastEditTime: 2020-07-18 18:09:35
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \SR\silency_test.py
'''

import pytest 
from spectral_residual import Silency
from srdata import data, score_normal
spec = Silency(amp_window_size=3, series_window_size=3, score_window_size=5)

@pytest.mark.parametrize(["values","expected"], [(data, score_normal)])
def test_generate_anomaly_score(values, expected):
    actual = spec.generate_anomaly_score(values)
    for ac, sc in zip(actual, expected):
        print(ac, sc)


actual = spec.generate_anomaly_score(data,type="abs")
print(data)
print(actual)
