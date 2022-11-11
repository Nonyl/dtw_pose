
import numpy as np
from realize import realize
import pandas as pd
from realize.realize_data import *

#x = np.array([0,2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
#y = np.array([0,1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
x = np.array([0,2,3,4,7,9,2,1,2,1]).reshape(-1, 1)
y = np.array([0,1,1,1,1,2,3,3,4,7,8,9,1,1,1,1]).reshape(-1, 1)
#datax = np.loadtxt('data1.txt',dtype=np.float,delimiter='\n   ')
#datay = np.loadtxt('data2.txt',dtype=np.float,delimiter='\n   ')
#datax = np.insert(datax,0,[0.0])
#datay = np.insert(datay,0,[0.0])
data1 = pd.read_csv('./data/output_stand.csv')
#data2 = pd.read_csv('./data/output2.csv')

neck_data = Coordinate_Neck(data1)
#print(neck_data)
neck_data.to_csv('./data/test0.csv', index=None)

#深复制，浅复制原来的值会被改变
df1 = neck_data.copy(deep=True)
gravity_data = center_gravity(df1)
print(gravity_data)
gravity_data.to_csv('./data/test1.csv', index=None)

#print(neck_data)
df2 = neck_data.copy(deep=True)

limb_angle_data = limb_angle(df2)
print(limb_angle_data)
limb_angle_data.to_csv('./data/test2.csv', index=None)

df3 = neck_data.copy(deep=True)
body_orient_data = body_orient(df3)
print(body_orient_data)
body_orient_data.to_csv('./data/test3.csv', index=None)

df4 = pd.concat([gravity_data,limb_angle_data,body_orient_data],axis=1)
print(df4)
df4.to_csv('./data/data_stand.csv', index=None)