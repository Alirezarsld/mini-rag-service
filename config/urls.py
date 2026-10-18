from django.urls import include, path

urlpatterns = [path("api/", include("rag.urls"))]
