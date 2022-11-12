import numpy as np
from realize import realize
import pandas as pd
from realize.realize import realize_dtw
from realize import survey
from realize.survey import *
#x = np.array([0,2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
#y = np.array([0,1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
x = np.array([0,2,3,4,7,9,2,1,2,1]).reshape(-1, 1)
y = np.array([0,1,1,1,1,2,3,3,4,7,8,9,1,1,1,1]).reshape(-1, 1)
#datax = np.loadtxt('data1.txt',dtype=np.float,delimiter='\n   ')
#datay = np.loadtxt('data2.txt',dtype=np.float,delimiter='\n   ')
#datax = np.insert(datax,0,[0.0])
#datay = np.insert(datay,0,[0.0])
datax = pd.read_csv('./data/data_after_output1.csv')
datay = pd.read_csv('./data/data_after_output3.csv')

realize_dtw(datax, datay)
#survey_data(datax, datay)
