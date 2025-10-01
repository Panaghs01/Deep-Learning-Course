import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse,AddNoise,Convolve
from tsaug.visualization import plot

days = 365
t = np.arange(days)

temps = 10 * np.sin(2 * np.pi * t / days)
avg_temp = 18 #GLOBAL WARMING!!
data = avg_temp + temps
noise = np.random.normal(0,0.3,size = days)

original = temps + data + noise
#original = original.reshape(1,-1)


augmenter = (
        AddNoise(scale=0.01)    #add small noise in the data
        + TimeWarp()*5      #compute 5 times in parallel all the transforms bellow
        + Convolve(window= 'flattop',size =11,prob=0.8)     #smoothen with kernel flattop
        + Crop(size= 330) #Crop the computed data to introduce some steep slopes
        + Drift(max_drift=0.2,n_drift_points=5,prob=0.5)
        + Quantize(n_levels=80,prob=0.2)
        + Reverse(prob= 0.2)
    )

y = augmenter.augment(original)

new = np.concatenate(y)

plt.figure(figsize=(26,16))
plt.plot(original)
plt.plot(y[0])
plt.plot(y[1])
#plt.plot(new)
plt.legend(['Original Data','1st augmentation','2nd augmentation'],fontsize='30')
plt.show()