import os

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from predict.views.lstm.model import LSTM
import time


# 定义滑动窗口函数
def sliding_window(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        window = data[i:(i + seq_length), 0]
        X.append(window)
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


# 使用股票数据进行训练，返回训练的结果并对股票进行保存
def train_lstm(stock_path, model=None, predictionTarget='Close', update_line_target=1, window=5):
    # 加载数据
    # 要从更新行进行加载，如果不需要更新，这个代码买会自动从第一行进行加载
    # 使用 skiprows 参数指定从某一行开始读取数据，这是增量学习的一部分
    data = pd.read_csv(stock_path, skiprows=update_line_target, header=None)
    with open(stock_path) as f:
        column_names = f.readline().strip().split(',')
    # 为DataFrame添加列
    data.columns = column_names
    data['Date'] = pd.to_datetime(data['Date'])

    # 将日期列转换为日期时间类型
    is_sorted = data['Date'].is_monotonic_increasing
    # 检查日期列是否递增
    if not is_sorted:
        # 如果日期列不是按照时间顺序排列，则对整个DataFrame进行逆序操作
        data = data.sort_values(by='Date', ascending=True)
        # 将逆序后的DataFrame写入新的CSV文件
        data.to_csv(stock_path, index=False)

    # 这里选择close列作为特征，并对其进行预测
    feature_data = data[predictionTarget].values.astype(float).reshape(-1, 1)

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_data_normalized = scaler.fit_transform(feature_data)

    # 设置滑动窗口大小和预测天数
    seq_length = window

    # 创建训练数据集和测试数据集
    if update_line_target == 1:
        train_size = int(len(feature_data_normalized) * 0.7)
        train_data, test_data = feature_data_normalized[0:train_size], feature_data_normalized[
                                                                      train_size:len(feature_data_normalized)]

        # 使用滑动窗口函数准备训练数据和测试数据
        X_train, y_train = sliding_window(train_data, seq_length)
        X_test, y_test = sliding_window(test_data, seq_length)

        # 将数据转换为PyTorch张量
        X_train = torch.from_numpy(X_train).type(torch.Tensor)
        X_test = torch.from_numpy(X_test).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        y_test = torch.from_numpy(y_test).type(torch.Tensor)

        X_train = X_train.unsqueeze(-1)
        X_test = X_test.unsqueeze(-1)
        y_train = y_train.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)
    else:
        train_size = int(len(feature_data_normalized) * 1)
        train_data, test_data = feature_data_normalized[0:train_size], feature_data_normalized[
                                                                       train_size:len(feature_data_normalized)]

        # 使用滑动窗口函数准备训练数据和测试数据
        X_train, y_train = sliding_window(train_data, seq_length)
        X_test, y_test = sliding_window(test_data, seq_length)

        # 将数据转换为PyTorch张量
        X_train = torch.from_numpy(X_train).type(torch.Tensor)
        X_test = torch.from_numpy(X_test).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        y_test = torch.from_numpy(y_test).type(torch.Tensor)

        X_train = X_train.unsqueeze(-1)
        X_test = X_test.unsqueeze(-1)
        y_train = y_train.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)

    # 初始化模型参数
    input_dim = 1
    hidden_dim = 16
    num_layers = 1
    output_dim = 1
    num_epochs = 500
    learning_rate = 0.001

    # 实例化模型、损失函数和优化器
    if update_line_target == 1:
        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)
    else:
        model = model
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练模型
    start_time = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        if epoch % 10 == 0:
            print("Epoch {}/{} - Loss: {}".format(epoch + 1, num_epochs, loss.item()))
        # send progress bar information
        progress = int((epoch / num_epochs) * 100)
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            'progress-bar-1',
            {
                'type': 'send.progress',
                'progress': progress,
            }
        )
    # 结束记录训练时间
    end_time = time.time()
    # 计算训练时间
    training_time_seconds = end_time - start_time
    training_time_minutes = training_time_seconds / 60  # 转换成分钟
    print("训练时间（秒）:", training_time_seconds)
    print("训练时间（分钟）:", training_time_minutes)

    # 获取当前工作目录
    current_directory = os.getcwd()
    print("当前工作目录:", current_directory)
    final_dir = f"{current_directory}/predict/views/lstm"

    # 保存模型
    stock_name = os.path.basename(stock_path)
    model_name = stock_name.replace('.csv', f'_{predictionTarget}.pth')
    torch.save(model.state_dict(), f'{final_dir}/model/{model_name}')

    # 返回模型
    return model
