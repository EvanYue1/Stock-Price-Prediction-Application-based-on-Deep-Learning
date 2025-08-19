from django.urls import path
from predict import consumers

websocket_urlpatterns = [
    # xxxx/progressbar/x1
    path('room/progressbar', consumers.ChatConsumer.as_asgi()),
]