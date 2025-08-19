from django.shortcuts import HttpResponse
from django.views.decorators.csrf import csrf_exempt  # 免除 csrf token检查
from predict.views.lstm_cnn_attention.application import lstm_cnn_se
from predict.views.lstm.application import lstm
import os
import json


# Create your views here.
@csrf_exempt
def predict(request):
    # 获取当前工作目录
    current_directory = os.getcwd()
    print("当前工作目录:", current_directory)

    # 初始化横坐标和预测结果
    x_axis = []
    y_axis = []

    combined_data = []
    combined_dates = []
    y_pred_inv = []
    y_true_inv = []
    combined_verify_data = [],
    combined_verify_ture_data = []
    combined_verify_dates = []
    """ predict """
    '''
        对于lstm
        这里的model_name并不是完整的model名字，
        而是最终model名字的第一部分，
        预测模型的名字一共有三部分，
        第一部分是股票代码，
        第二部分是预测目标
        第三部分是后缀.pth
    '''
    '''
        对于LSTM-CNN-Attention
        这里的model_name并不是完整的model名字，
        而是最终model名字的第一部分，
        预测模型的名字一共有三部分，
        第一部分是股票代码，
        第二部分是预测目标
        第三部分是后缀.pth
    '''
    # judge the request model
    if request.method == 'POST':
        stock_symbol = request.POST.get('stockSymbol')
        stock_code = stock_symbol
        predictionTarget = request.POST.get('predictionTarget')  # 获取预测目标
        # selectedFeatures = request.POST.get('selectedFeatures')  # 获取预测使用的特征
        selectedFeatures = request.POST.getlist('selectedFeatures[]', [])
        selectedFeatures_simply_name = ''.join(sorted(word.title()[0] for word in selectedFeatures))
        selectedFeatures = [feature.capitalize() for feature in selectedFeatures]

        # 需要对传来的参数做一些预处理
        first_space_index = predictionTarget.find(' ')
        if first_space_index != -1:
            predictionTarget = predictionTarget[:first_space_index]
        else:
            predictionTarget = predictionTarget
        predictionTarget = predictionTarget.capitalize()
        # 检查文件名是否已经以.txt结尾，如果没有则添加
        if not stock_symbol.endswith(".csv"):
            stock_symbol += ".csv"

        data_path = os.path.join(current_directory, "predict/dataset/{}".format(stock_symbol))
        future_days = request.POST.get('predictionDays')
        model_type = request.POST.get('model')
        future_days = int(future_days)
        if model_type == 'LSTM_CNN_CBAM' or model_type == 'LSTM_CNN_Base':
            print(f"OK, model type is {model_type}")
            model_type = model_type.split('_')[2]
            model_name = f"{model_type}_{stock_code}_{predictionTarget}_best#{selectedFeatures_simply_name}.pth"
            print("model name:", model_name)
            print("data_path:", data_path)
            print("future_days:", future_days)
            combined_data, combined_dates, y_pred_inv, y_true_inv, combined_verify_data, combined_verify_ture_data, combined_verify_dates = lstm_cnn_se(
                model_name=model_name,
                data_path=data_path,
                selectedFeatures=selectedFeatures,
                future_days=future_days,
                predictionTarget=predictionTarget)
        elif model_type == 'LSTM':
            print("OK, model type is LSTM")
            model_name = f"{stock_code}_{predictionTarget}.pth"  # the name of this model still lack the .pth suffix
            print("model name:", model_name)
            print("data_path:", data_path)
            print("future_days:", future_days)
            combined_data, combined_dates, y_pred_inv, y_true_inv, combined_verify_data, combined_verify_ture_data, combined_verify_dates = lstm(model_name=model_name, data_path=data_path,
                                                                         future_days=future_days,
                                                                         predictionTarget=predictionTarget)
        else:
            print("OK, model type is RNN")
    # 构造返回的数据
    response_data = {
        'combined_data': combined_data,
        'combined_dates': combined_dates,
        'y_pred_inv': y_pred_inv,
        'y_true_inv': y_true_inv,
        'combined_verify_data': combined_verify_data,
        'combined_verify_ture_data': combined_verify_ture_data,
        'combined_verify_dates': combined_verify_dates
    }
    return HttpResponse(json.dumps(response_data), content_type="application/json")
