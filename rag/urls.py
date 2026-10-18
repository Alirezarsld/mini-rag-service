from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.HealthView.as_view()),
    path("documents/", views.DocumentListCreateView.as_view()),
    path("documents/upload/", views.DocumentUploadView.as_view()),
    path("documents/<int:pk>/", views.DocumentDetailView.as_view()),
    path("search/", views.SearchView.as_view()),
    path("ask/", views.AskView.as_view()),
]
