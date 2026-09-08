"""
URL routing for the object_detection app.
Maps web addresses to the view functions that handle them.
"""
from django.urls import path
from .views import upload_view, result_view

urlpatterns = [
    # Homepage: the upload form.
    path("", upload_view, name="upload"),

    # Result page for a single prediction, e.g. /result/3/
    path("result/<int:pk>/", result_view, name="result"),
]