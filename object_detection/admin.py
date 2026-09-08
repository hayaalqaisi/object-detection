"""
Register our model with Django's admin panel so you can browse predictions
at /admin/ (after creating a superuser).
"""
from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("id", "label", "confidence_percent", "created_at")
    list_filter = ("label",)
