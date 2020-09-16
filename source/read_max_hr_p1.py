import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy import stats
import numpy as np

# file_name = '../data/user01.csv'  # 这里烦劳志强写一下，根据健康表选择页面传过来的选中后的表格名字，确保能读入
# data = pd.read_csv(file_name)

#定义字符串分割&&取最大值函数
def split_text(Series_x):
    max_values = []
    for ii in range(Series_x.shape[0]):
        x = Series_x.loc[ii].replace('--','')
        x = x.replace('-','')
        x = x.split(',')
        x = np.array(x)
        x[x == '']='0'
        #防止出现过大的数据
        x = x.astype(np.int32)
        big_values = x[x > 120]
        big_values = np.mod(big_values,200)
        x[x > 120] = big_values
        max_x = np.max(x.astype(np.int32))
        max_values.append(max_x)
    max_col = np.max(max_values)
    ind_max_row = np.argmax(max_values)
    return max_col,str(max_col) ,ind_max_row

# 这里还要烦劳志强写一下
# 其中
# max_HR 为从表格读取的数值型最大心率
# str_max_HR 为从表格读取的字符型最大心率
# max_HR, str_max_HR, argmax = split_text(data['liveHR'])
# print(max_HR)
# print(str_max_HR)










