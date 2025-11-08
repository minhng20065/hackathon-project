import numpy as np
import torch
import pandas as pd
print("test")

data1 = pd.read_csv("C:/Users/bsche/OneDrive/Desktop/Dev/PhishingDetection/archive/CEAS_08.csv")
print(data1.head())
data1["Sender Name"] = data1["sender"].str.split("<").str[0]
data1["Sender Email"] = data1["sender"].str.split("<").str[1]
data1["Sender Email"] = data1["Sender Email"].str.split(">").str[0]
print(data1.head())
data1 = data1.drop(columns=["sender", "date", "receiver"])
print(data1.head().T)

data2 = pd.read_csv("archive/Enron.csv")
print(data2.head().T)