from channels.generic.websocket import WebsocketConsumer
from channels.exceptions import StopConsumer
from asgiref.sync import async_to_sync
import json


class ChatConsumer(WebsocketConsumer):
    def websocket_connect(self, message):
        print("OK connect")
        # receive the client connect
        self.accept()  # accept() function means that the server allow the client to connect

        # put this client connection object in some place(in memory or redis)
        async_to_sync(self.channel_layer.group_add)("progress-bar-1", self.channel_name)

    def websocket_receive(self, message):
        # the browser send message to the server, this function will be executed
        pass

    def send_progress(self, event):
        progress = event['progress']
        self.send(text_data=json.dumps({
            'progress': progress
        }))

    def websocket_disconnect(self, message):
        async_to_sync(self.channel_layer.group_discard)("progress-bar-1", self.channel_name)
        # when the client disconnect actively, this function be executed
        raise StopConsumer()
