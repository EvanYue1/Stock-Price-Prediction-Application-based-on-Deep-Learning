# import torch
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from predict.views.lstm.dataloader import DataGenerator
#
#
# def make_test_data(df, test_size=0.9):
#     # 计算CSV文件的总行数
#     total_rows = len(df)
#     # 计算后20％的行数范围
#     start_row = int(total_rows * (1 - test_size))
#     print("test start row: ", start_row)
#     end_row = total_rows
#     print("test end row: ", end_row)
#     # 读取后20％的数据
#     last_20_percent_data = df.iloc[start_row:end_row]
#     print("the start test row data is: ", last_20_percent_data.iloc[0])
#     print("the end test row data is: ", last_20_percent_data.iloc[-1])
#     return last_20_percent_data
#
#
# def test(model, data_path, predictionTarget='Close', window=5):
#     # 加载数据
#     data = DataGenerator(data_path)
#     df = make_test_data(data.get_df())
#     df = df[predictionTarget].values.astype(float).reshape(-1, 1)
#     # 数据归一化
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     close_data_normalized = scaler.fit_transform(df)
#     min_value = scaler.data_min_
#     max_value = scaler.data_max_
#     print(f"正则化参数为: {scaler.data_min_}, {scaler.data_max_}")
#
#     # 记录data
#     date = []
#
#     # 定义滑动窗口函数
#     def sliding_window(data, seq_length):
#         X, y = [], []
#         for i in range(len(data) - seq_length - 1):
#             window = data[i:(i + seq_length), 0]
#             X.append(window)
#             y.append(data[i + seq_length, 0])
#         return np.array(X), np.array(y)
#
#     # 设置滑动窗口大小和预测天数
#     seq_length = window
#     x, y_true = sliding_window(close_data_normalized, seq_length)
#     x = torch.Tensor(x)
#     y_true = torch.Tensor(y_true)
#
#     x = x.unsqueeze(-1)
#     y_true = y_true.unsqueeze(-1)
#     criterion = torch.nn.MSELoss(reduction='mean')
#     # 模型评估
#     model.eval()
#     with torch.no_grad():
#         y_pred = model(x)
#         test_loss = criterion(y_pred, y_true)
#         print('Test Loss: {:.6f}'.format(test_loss.item()))
#
#     # 反归一化预测值
#     print(f"正则化参数为-: {scaler.data_min_}, {scaler.data_max_}")
#     y_true_inv = y_true.detach().numpy() * (max_value - min_value) + min_value
#     y_true_inv = np.squeeze(y_true_inv)
#     y_pred_inv = y_pred.detach().numpy() * (max_value - min_value) + min_value
#     y_pred_inv = np.squeeze(y_pred_inv)
#
#     # 计算方向准确度
#     correct_predictions = np.sum(np.sign(y_pred_inv[1:] - y_pred_inv[:-1]) == np.sign(y_true_inv[1:] - y_true_inv[:-1]))
#     total_predictions = len(y_pred_inv) - 1
#     direction_accuracy = correct_predictions / total_predictions
#     print("方向准确性: ", direction_accuracy)
#
#     # 计算RMSE
#     test_rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
#     print('Test RMSE: {:.6f}'.format(test_rmse))
#     return y_pred_inv, y_true_inv


import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from predict.views.lstm.dataloader import DataGenerator


def make_test_data(df, test_size=0.9):
    # 计算CSV文件的总行数
    total_rows = len(df)
    # 计算后20％的行数范围
    start_row = int(total_rows * (1 - test_size))
    print("test start row: ", start_row)
    end_row = total_rows
    print("test end row: ", end_row)
    # 读取后20％的数据
    last_20_percent_data = df.iloc[start_row:end_row]
    print("the start test row data is: ", last_20_percent_data.iloc[0])
    print("the end test row data is: ", last_20_percent_data.iloc[-1])
    return last_20_percent_data


def test(model, data_path, predictionTarget='Close', window=5):
    # 加载数据
    data = DataGenerator(data_path)
    df = data.get_df()[predictionTarget].values.astype(float).reshape(-1, 1)
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_data_normalized = scaler.fit_transform(df)
    min_value = scaler.data_min_
    max_value = scaler.data_max_
    print(f"正则化参数为: {scaler.data_min_}, {scaler.data_max_}")

    # 记录data
    date = []

    # 定义滑动窗口函数
    def sliding_window(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length - 1):
            window = data[i:(i + seq_length), 0]
            X.append(window)
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)

    # 设置滑动窗口大小和预测天数
    seq_length = window
    x, y_true = sliding_window(close_data_normalized, seq_length)
    # 制作测试集
    total_row = len(x)
    test_row = round(0.9 * total_row)
    x = x[int(test_row):]
    y_true = y_true[int(test_row):]
    print("test start row: ", test_row)
    print("test start x data: ", x[0])
    print("test start y data: ", y_true[0])
    x = torch.Tensor(x)
    y_true = torch.Tensor(y_true)

    x = x.unsqueeze(-1)
    y_true = y_true.unsqueeze(-1)
    criterion = torch.nn.MSELoss(reduction='mean')
    # 模型评估
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        test_loss = criterion(y_pred, y_true)
        print('Test Loss: {:.6f}'.format(test_loss.item()))

    # 反归一化预测值
    print(f"正则化参数为-: {scaler.data_min_}, {scaler.data_max_}")
    y_true_inv = y_true.detach().numpy() * (max_value - min_value) + min_value
    y_true_inv = np.squeeze(y_true_inv)
    y_pred_inv = y_pred.detach().numpy() * (max_value - min_value) + min_value
    y_pred_inv = np.squeeze(y_pred_inv)

    # 计算方向准确度
    correct_predictions = np.sum(np.sign(y_pred_inv[1:] - y_pred_inv[:-1]) == np.sign(y_true_inv[1:] - y_true_inv[:-1]))
    total_predictions = len(y_pred_inv) - 1
    direction_accuracy = correct_predictions / total_predictions
    print("方向准确性: ", direction_accuracy)

    # 计算RMSE
    test_rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    print('Test RMSE: {:.6f}'.format(test_rmse))
    return y_pred_inv, y_true_inv
