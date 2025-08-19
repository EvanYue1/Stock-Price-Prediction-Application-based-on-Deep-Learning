from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt  # 免除 csrf token检查

@csrf_exempt
def index(request):
    return render(request, "predict_main.html")