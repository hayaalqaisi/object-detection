"""App configuration. Django reads this to register the app."""
from django.apps import AppConfig


class ObjectDetectionConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "object_detection"
    verbose_name = "Object Detection (Cat vs Dog)"