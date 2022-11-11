#看那一列的差值会比较大
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
def survey_data(x,y):

    Num_col = x.shape[1]
    lenx = x.shape[0]+1
    leny = y.shape[0]+1
    print('num_col',Num_col)
    print('lenx:',lenx)
    print('leny:', leny)
    lens = min(leny,lenx)
    a = [0.0 for x in range(0, Num_col+1)]
    print(a)
    for i in range(lens-1):
        #print(i)
        for j in range(Num_col):
            a[j] = a[j] + ((float(x.iloc[i][j]) - float(y.iloc[i][j]))**2)
    for i in range(Num_col):
        if i!=0 and i%5 == 0:
            print()
        print(a[i], end='|')
def data_amplification(data):
    col = data.shape[0]

    df_1 = data['angle01']
    df_2 = data['angle02']
    df_3 = data['angle03']
    df_4 = data['angle04']
    df_5 = data['angle05']
    df_6 = data['angle06']
    df_7 = data['angle07']
    df_8 = data['angle08']
    df_9 = data['angle09']
    df_10 = data['angle10']
    df_11 = data['angle11']
    df_12 = data['angle12']
    df_13 = data['angle13']
    #df_1 = data['limb_angle01']

    for i in range(col):
        df_1[i] = df_1[i] * 3
        df_2[i] = df_2[i] * 3
        df_3[i] = df_3[i] * 3
        df_4[i] = df_4[i] * 3
        df_5[i] = df_5[i] * 3
        df_6[i] = df_6[i] * 3
        df_7[i] = df_7[i] * 3
        df_8[i] = df_8[i] * 3
        df_9[i] = df_9[i] * 3
        df_10[i] = df_10[i] * 3
        df_11[i] = df_11[i] * 3
        df_12[i] = df_12[i] * 3
        df_13[i] = df_13[i] * 3

    data['angle01'] = df_1
    data['angle02'] = df_2
    data['angle03'] = df_3
    data['angle04'] = df_4
    data['angle05'] = df_5
    data['angle06'] = df_6
    data['angle07'] = df_7
    data['angle08'] = df_8
    data['angle09'] = df_9
    data['angle10'] = df_10
    data['angle11'] = df_11
    data['angle12'] = df_12
    data['angle13'] = df_13


    return data

if __name__ == '__main__':

    data = pd.read_csv('E:/学习/python/python code/dtw_pose/data/data_stand.csv')
    data = data_amplification(data)
    data.to_csv('E:/学习/python/python code/dtw_pose/data/data_amp_stand.csv', index=None)