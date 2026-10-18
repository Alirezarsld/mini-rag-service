from django.urls import include, path

from rag.views import home

urlpatterns = [
    path("", home),
    path("api/", include("rag.urls")),
]
