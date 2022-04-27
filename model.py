import numpy as np
import pandas as pd
import math as m
from sklearn.linear_model import LogisticRegression
import pickle

def rescale_values(col_name, scale):
    original_col = dataset[col_name]
    rescaled_col = original_col.apply(scale)
    return rescaled_col

def rename_col(original_name, new_name):
    dataset[new_name] = dataset[original_name]
    dataset.drop([original_name], axis = 1, inplace = True)

dataset = pd.read_csv('stardataset.csv')
dataset.drop(["Star color"], axis = 1, inplace = True)

dataset["Radius(R/Ro)"] = rescale_values("Radius(R/Ro)", m.log10)
dataset["Luminosity(L/Lo)"] =  rescale_values("Luminosity(L/Lo)", m.log10)

rename_col("Luminosity(L/Lo)", "log(L/Lo)")
rename_col("Radius(R/Ro)", "log(R/Ro)")

#Use mapping to convert spectral class from a letter to a number
sp_class={"O":0,"B":1,"A":2,"F":3,"G":4,"K":5,"M":6}
dataset["Spectral Class"] = dataset["Spectral Class"].map(sp_class)

y = dataset["Star type"]
X = dataset.drop(["Star type"], axis = 1)

dataset.describe().drop(["Star type"], axis = 1).to_csv("datasetstatistics.csv", header = [0,1,2,3], index = False, columns=["Temperature (K)", "log(L/Lo)", "log(R/Ro)", "Absolute magnitude(Mv)"])

print(dataset.describe().drop(["Star type", "Spectral Class"], axis = 1))
lr = LogisticRegression(max_iter=10000)
lr.fit(X,y)

pickle.dump(lr, open('model.pkl','wb'))
