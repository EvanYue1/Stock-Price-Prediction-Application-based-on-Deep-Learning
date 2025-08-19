import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class DataGenerator(Dataset):
    def __init__(self, data_path='', predictionTarget='Close'):
        df1 = pd.read_csv(data_path, parse_dates=['Date'])  # 加载数据
        self.df = df1
        df1 = df1[predictionTarget].values.astype(float).reshape(-1, 1)
        # 数据归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = scaler
        data_normalized = scaler.fit_transform(df1)
        self.data = data_normalized

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def get_df(self):
        return self.df

    def get_scaler(self):
        return self.scaler
