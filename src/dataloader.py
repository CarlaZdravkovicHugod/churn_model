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

    # TODO:
    # def visualize(self, idx: int) -> None:
    #     if self.data is None: self.load_data()
    #     plt.figure(figsize=(20, 10))
    #     plt.plot(self.data.iloc[idx])
    #     plt.show

    # def __len__(self) -> int: return len(self.data)
    # def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]: return self.data[idx]

if __name__ == "__main__":
    dataset = ChurnDataset()
    print(dataset.get_data())
    dataset.load_data()
    print('Lenght of data set:', len(dataset.data))
    # dataset.visualize(0)