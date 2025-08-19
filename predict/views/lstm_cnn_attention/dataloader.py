import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset


class DataGenerator(Dataset):

    def __init__(self,
                 data_path,
                 selectedFeatures,
                 predictionTarget,
                 isPredict=True,
                 isTest=False,
                 window=5,
                 update_line_site=1):

        df1 = pd.read_csv(data_path, skiprows=update_line_site, header=None)
        with open(data_path) as f:
            column_names = f.readline().strip().split(',')
        # 为 DataFrame 添加列名
        df1.columns = column_names
        df1['Date'] = pd.to_datetime(df1['Date'])
        self.df = df1
        if isPredict is True:
            df1 = df1[selectedFeatures]
            column_index = self.df.columns.get_loc(predictionTarget)
            self.min_val_close = self.df.iloc[:, column_index].min()
            self.max_val_close = self.df.iloc[:, column_index].max()
            self.min_max_scaler = preprocessing.MinMaxScaler()
            df0 = self.min_max_scaler.fit_transform(df1)
            df = pd.DataFrame(df0, columns=df1.columns)
            result = np.array(df)
            self.data = result
        else:
            # 制作训练集数据
            """
                this flag represent whether the predictionTarget is included in the selectedFeatures, 
                if it is in, 
                then this 
                flag is true, 
                if not, 
                then is is false
            """
            # flag = True
            # if predictionTarget not in selectedFeatures:
            #     selectedFeatures.append(predictionTarget)
            #     flag = False
            df1 = df1[selectedFeatures]
            column_index = self.df.columns.get_loc(predictionTarget)
            self.min_val_close = self.df.iloc[:, column_index].min()
            self.max_val_close = self.df.iloc[:, column_index].max()
            self.min_max_scaler = preprocessing.MinMaxScaler()
            df0 = self.min_max_scaler.fit_transform(df1)
            df = pd.DataFrame(df0, columns=df1.columns)

            stock = df
            # 获取目标列位于第几列
            predictionTarget_col_index = stock.columns.get_loc(predictionTarget)
            seq_len = window
            amount_of_features = len(stock.columns)  # 有几列
            data = stock.values  # pd.DataFrame(stock) 表格转化为矩阵
            sequence_length = seq_len + 1  # 序列长度
            result = []
            for index in range(len(data) - sequence_length):  # 循环数据长度-sequence_length次
                result.append(data[index: index + sequence_length])  # 第i行到i+sequence_length
            result = np.array(result)  # 得到样本，样本形式为6天*3特征
            row = round(0.9 * result.shape[0])  # 划分训练集测试集
            train = result[:int(row), :]
            x_train = train[:, :-1]
            y_train = train[:, -1][:, predictionTarget_col_index]
            x_test = result[int(row):, :-1]
            print("test start row: ", int(row))
            print("test start x data: ", x_test[0])
            y_test = result[int(row):, -1][:, predictionTarget_col_index]
            print("test start y data: ", y_test[0])
            # reshape成 6天*3特征
            X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
            X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
            if not isTest:
                self.data = X_train
                self.label = y_train
            else:
                self.data = X_test
                self.label = y_test

    def get_scaler(self):
        return self.min_max_scaler

    def get_min_max_val_close(self):
        return self.min_val_close, self.max_val_close

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def get_df(self):
        return self.df

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).to(torch.float32), torch.FloatTensor([self.label[idx]])


