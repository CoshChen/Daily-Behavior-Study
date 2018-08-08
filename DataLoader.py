# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:44:46 2018

@author: Ko-Shin Chen
"""

import csv
import numpy as np

input_file = './ProcessedData/Final_Study_1_Study_2_Merged_input.csv'

with open(input_file) as in_data:
    reader = csv.DictReader(in_data)
    
    PANA = np.zeros([353, 11, 2])
    Exe = np.zeros([353, 11, 3])
    DHQ = np.zeros([353, 11, 20])
    Base = np.zeros([353, 4])
    
    for row in reader:
        day = int(row['Day'])
        sid = int(row['SID'])
        
        if day == 1:
            Base[sid,0] = float(row['DASS_dep'])
            Base[sid,1] = float(row['DASS_anx'])
            Base[sid,2] = float(row['DASS_strs'])
            Base[sid,3] = float(row['CAMS'])
        
        PANA[sid,11-day,0] = float(row['PA'])
        PANA[sid,11-day,1] = float(row['NA'])
        
        Exe[sid,11-day,0] = float(row['LTEQ_strenuous'])
        Exe[sid,11-day,1] = float(row['LTEQ_moderate'])
        Exe[sid,11-day,2] = float(row['LTEQ_minimal'])
        
        DHQ[sid,11-day,0] = float(row['Alcohol'])
        
        for i in range(1,20):
            DHQ[sid,11-day,i] = float(row['DHQ'+str(i)])
        

outfile = './ProcessedData/input.npz'
np.savez(outfile, PANA=PANA, Exe=Exe, DHQ=DHQ, Base=Base)
            



