"""
ASGI config for stock_forecast_djangoWeb_version1 project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from . import routings

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE", "stock_forecast_djangoWeb_version1.settings"
)

# application = get_asgi_application()
application = ProtocolTypeRouter({
    "http": get_asgi_application(),         # automatically find urls.py and then find the views function   --> http
    "websocket": URLRouter(routings.websocket_urlpatterns)    # creating the routings is the same as creating urls、creating the consumers is the same as views function
})
