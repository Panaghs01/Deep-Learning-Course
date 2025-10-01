# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:50:34 2025

@author: panos
"""

import pandas as pd

dictionary = {'TF':'1','airplane':'1','cat':'0',
              'dog':'0','fruit':'0','person':'0','motorbike':'0','flower':'0','car':'0'}

df = pd.read_csv('image_labels.csv')

df.replace({'label':{'TF':'1','airplane':'1','cat':'0','dog':'0','fruit':'0','person':'0','motorbike':'0','flower':'0','car':'0'}},inplace=True)

df.to_csv('./labels.csv',index=False)