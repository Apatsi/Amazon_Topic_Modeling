# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
import os

print(os.getcwd())

# Display all of the files found in your current working directory
print(os.listdir(os.getcwd()))

data = pd.read_csv('C:/Users/Akisp/PycharmProjects/TEST/dataset/pokemon.csv')
print(data.head(), "\n")

data.info()


print("\n", data.describe())

minHP1 = data.iloc[data.index[data['HP'] == data["HP"].min()]]
minHP2 = data.loc[data["HP"].values == data["HP"].min()]

print("\n", minHP1)
print("\n", minHP2)

skata = (data["HP"].values > data["HP"].min())
print(skata)

#Otan kanoume search mesa sto dataframe gia na vroume times to kanoyme me logikes pules kai .values!!!
minHP3 = np.logical_and(data["Attack"].values >= 150, data["Speed"].values >= 150)
print("\n", minHP3)

print(data["HP"].values)

data1 = data[minHP3]
print(data1)

#Stin .loc aristera einai ta indexes kai deksia einai ta features!!!
print(data.loc[0:5, ["Name", "HP", "Attack"]])

# Mporoume na antikatastisoume to INDEX me features tou dataframe mas!! Anti diladi gia to default 0,1,2... bazoume p.x. ta TYPES!!!
data1 = data.set_index(["Type 1", "Type 2"])
print(data1)

# To .groupby("Feature") ousiastika pairnei to feature kai to kanei INDEX kai meta panw se auto kanoume .mean,.min,.max,.median, klp!!!
data2 = data.groupby("Type 1").mean()
print(data2)
data2.info()

