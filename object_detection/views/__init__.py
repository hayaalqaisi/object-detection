# Makes "views" a package and re-exports each view so urls.py can write:
#   from object_detection.views import upload_view, result_view
from .upload_view import upload_view
from .result_view import result_view

__all__ = ["upload_view", "result_view"]