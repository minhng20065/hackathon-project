import numpy as np
import torch
import pandas as pd
# print("test")

CEAS_08 = pd.read_csv("hackathon-project/CEAS_08.csv")
Nigerian_Fraud = pd.read_csv("hackathon-project/Nigerian_Fraud.csv")
dataList = [CEAS_08, Nigerian_Fraud]
for data in dataList:
    data["Sender Name"] = data["sender"].str.split("<").str[0]
    data["Sender Email"] = data["sender"].str.split("<").str[1]
    data["Sender Email"] = data["Sender Email"].str.split(">").str[0]
    data.drop(columns=["sender", "date", "receiver"], inplace=True)
    # print(f"data: {data.head().T}") #make something that goes through body if necessary
print(CEAS_08.columns)

# pd.set_option("display.width", None)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_colwidth", None)
# print(Nigerian_Fraud["body"].head())
# pd.reset_option("display.width")
# pd.reset_option("display.max_columns")
# pd.reset_option("display.max_colwidth")


