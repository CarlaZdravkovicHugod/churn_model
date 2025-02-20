import kagglehub
import os
from torch.utils.data import Dataset
import pandas as pd


class ChurnDataset(Dataset):
    def __init__(self):
        self.path = kagglehub.dataset_download("blastchar/telco-customer-churn")

        self.data = None

    def get_data(self) -> None: return self.path

    def load_data(self, reload: bool = False) -> None:
        if not reload and self.data is not None: return
        with open(os.path.join(self.path,os.listdir(self.path)[1])) as f:
            df = pd.read_csv(f)

        self.data = df
        print('Data loaded')
        return self.data

    def preprocess_data(self) -> None:
        self.data.drop(columns=['customerID'])

        # TODO: Map all columns where the only answers are yes and no to 1 and 0
        #self.data.replace(to_replace={'Yes':1, 'No':0, 'Male':1, 'Female':0})
        # self.data['Churn'] = self.data['Churn'].map({'Yes': 1, 'No': 0})

        self.data = pd.get_dummies(self.data, columns=['PaymentMethod'])
        self.data.renamed(columns=lambda x: x.replace("PaymentMethod_", ""), inplace=True)
        one_hot_columns = self.data.select_dtypes(include=["bool"]).columns
        self.data[one_hot_columns] = self.data[one_hot_columns].astype(int)
        self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')

        # Define X and y

if __name__ == "__main__":
    dataset = ChurnDataset()
    print(dataset.get_data())
    dataset.load_data()
    print('Lenght of data set:', len(dataset.data))
    # dataset.visualize(0)