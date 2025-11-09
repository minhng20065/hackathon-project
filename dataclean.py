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

enron_data = pd.read_csv("enron_data_fraud_labeled.csv", usecols = ["From", "Subject", "X-From", "Body", "Label", "To"])
# print(enron_data.head().T)
# enron_data = enron_data.drop(columns = ["Folder-User", "Folder-Name", "Message-ID", "Date", "To", "Mime-Version", "Content-Type", "Content-Transfer-Encoding", "X-To", "X-cc", "X-bcc", "X-Folder", "X-Origin", "X-FileName", "Cc", "Bcc", "Time", "Attendees", "Re", "Source", "Mail-ID", "POI-Present", "Suspicious-Folders", "Sender-Type", "Unique-Mails-From-Sender", "Low-Comm", "Contains-Reply-Forwards"])
#Kept: From, Subject, X-From, Body, Label
#Time is blank. IDK what POI is

# enron_data = enron_data[["From", "Subject", "X-From", "Body", "Label", "To"]]
enron_data["X-From"] = enron_data["X-From"].str.split('"').str[1]

# print(f"enron_data: {enron_data.shape}")
# for i, item in enumerate(reversed(enron_data["From"])):
#     if enron_data["From"][i] == enron_data["To"][i]:
#         enron_data = enron_data.drop([i]) #IDK how this works, but it will either go through the whole dataset to drop O(n), or just delete it O(1). Because this goes through the whole list, it would be times O(n)
# data.drop([i]) creates a new dataframe each time. So it will take O(n) time no matter how you traverse it
mask = enron_data["To"] != enron_data["From"]
enron_data = enron_data[mask]
# print("~13,000 self emails deleted!")

enron_data = enron_data.drop(columns= ["To"])
# print(enron_data.head().T)

# print()

# print(Nigerian_Fraud.columns, enron_data.columns)
# Renaming and reordering the columns to have consistent formatting
enron_data["hasURL"] = pd.NA
enron_data = enron_data[["X-From", "From", "Subject", "Body", "hasURL", "Label"]]
CEAS_08 = CEAS_08[["Sender Name", "Sender Email", "subject", "body", "urls", "label"]]
Nigerian_Fraud = Nigerian_Fraud[["Sender Name", "Sender Email", "subject", "body", "urls", "label"]]
# print(enron_data.head().T)
# print(Nigerian_Fraud.columns, enron_data.columns)

# print("\nFor loop:")

datasets = [CEAS_08, Nigerian_Fraud, enron_data]
megaDataset = pd.DataFrame({})
for data in datasets:
    data.columns = ["Sender Name", "Sender Email", "Subject", "Body", "hasURL", "isSpam"]
    # print(f"{data} \ncolumns: {data.columns}, shape: {data.shape}\n")
megaDataset = pd.concat(datasets)
# megaDataset2 = pd.concat(datasets, ignore_index=True) # returns the same .shape
print(megaDataset.T)
# print(f"shape: {megaDataset.shape}")
# print("Why is the columns in 'megaDataset.T' and 'megaDataset.shape' different? Maybe if theres not enough memory allocation available. I just restarted the terminal and VS Code")
