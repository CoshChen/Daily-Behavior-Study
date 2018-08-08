# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:09:35 2018

@author: Ko-Shin Chen
"""

import numpy as np
import pandas as pd


D = 11 # days threshold
csv_file = './Dataset/Final_Study_1_Study_2_Merged.csv'
processed_csv_file = './ProcessedData/Final_Study_1_Study_2_Merged_processed.csv'

full_data = pd.read_csv(csv_file)
"""
Column Names:

ID_merged, ID, Day, Age, Gender, Ethnicity, Race
DASS1, DASS2, DASS3, DASS4, DASS5, DASS6, DASS7, DASS8, DASS9, DASS10, DASS11, 
DASS12,DASS13, DASS14, DASS15, DASS16, DASS17, DASS18, DASS19, DASS20, DASS21
CAMS1, CAMS2, CAMS3, CAMS4, CAMS5, CAMS6, CAMS7, CAMS8, CAMS9, CAMS10, CAMS11, CAMS12

Alcohol, Smoke
DHQ1, DHQ2, DHQ3, DHQ4, DHQ5, DHQ6, DHQ7, DHQ8, DHQ9, DHQ10, DHQ11, DHQ12
DHQ13, DHQ14, DHQ15, DHQ16, DHQ17, DHQ18, DHQ19
LTEQ_minimal, LTEQ_moderate, LTEQ_strenuous

PANAS1, PANAS2, PANAS3, PANAS4, PANAS5, PANAS6, PANAS7, PANAS8, PANAS9, PANAS10,
PANAS11, PANAS12, PANAS13, PANAS14, PANAS15, PANAS16, PANAS17, PANAS18, PANAS20

PrimaryFirst, DASS_dep, DASS_anx, DASS_strs, PA, NA

(SYSMISS = 999)
"""


data = full_data.drop(columns=['ID']).rename(columns={'ID_merged': 'ID'})
# drop 'ID' and rename 'ID_merged' as 'ID'
data[data.columns] = data[data.columns].apply(pd.to_numeric)


"""
Drop samples that has not enough records (D days).  
"""
ID_days = data.groupby(['ID']).size()
ID_keep = ID_days[ID_days >= D].index

data = data[data['ID'].isin(ID_keep)] 
# drop samples having records less than D days

sample_size = len(ID_keep)
print('Sample Size = ' + str(sample_size))

"""
Replacing Missing Value
"""
DailyHB = ['DHQ'+str(i) for i in range(1,20)]
data[data[DailyHB]==999] = 0 # replace missing value by 0: zero times
LETQ = ['LTEQ_minimal', 'LTEQ_moderate', 'LTEQ_strenuous'] 
data[data[LETQ]==999] = 0 # replace missing value by 0: zero times

data[data[['Alcohol']]==999] = 0
data[data[['Smoke']]==999] = 0 


"""
Summarize DASS
"""
DASS = ['DASS'+str(i) for i in range(1,22)]
DASS_dep = ['DASS'+str(i) for i in [3,5,10,13,16,17,21]]
DASS_anx = ['DASS'+str(i) for i in [2,4,7,9,15,19,20]]
DASS_strs = ['DASS'+str(i) for i in [1,6,8,11,12,14,18]]

data[data[DASS]==999] = 0 # replace missing value by 0: not at all

data['DASS_dep'] = data[DASS_dep].sum(axis=1)
data['DASS_anx'] = data[DASS_anx].sum(axis=1)
data['DASS_strs'] = data[DASS_strs].sum(axis=1)


"""
Summarize CAMS
"""
CAMS = ['CAMS'+str(i) for i in range(1,13)]
CAMS_neg = ['CAMS2', 'CAMS6', 'CAMS7']
data[data[CAMS]==999] = 1 # replace missing value by 1: not at all

data['CAMS'] = data[CAMS].sum(axis=1) - 2*data[CAMS_neg].sum(axis=1)


"""
Summarize Positive/Negative Affect
"""
positive = ['PANAS'+str(i) for i in [1,3,5,9,10,12,14,16,17]]
negative = ['PANAS'+str(i) for i in [2,4,6,7,8,11,13,15,18,20]]

data[data[positive]==999] = 1 # replace missing value by 1: not at all
data[data[negative]==999] = 1

data['PA'] = data[positive].sum(axis=1)
data['NA'] = data[negative].sum(axis=1)


"""
Assign a Color for Each Sample 
"""
color_dict = {}
for id_num in ID_keep:
    color_dict[id_num] = np.random.rand(3)

new_col_1 = data['ID'].apply(lambda x: color_dict[x][0])
new_col_2 = data['ID'].apply(lambda x: color_dict[x][1])
new_col_3 = data['ID'].apply(lambda x: color_dict[x][2])

data.insert(loc=1, column='Color R', value=new_col_1)
data.insert(loc=2, column='Color G', value=new_col_2)
data.insert(loc=3, column='Color B', value=new_col_3)


"""
Export Processed Data
"""
data.to_csv(processed_csv_file)