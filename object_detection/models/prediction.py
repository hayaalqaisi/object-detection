

from django.db import models


class Prediction(models.Model):
    # The image the user uploaded. Files go into media/uploads/.
    image = models.ImageField(upload_to="uploads/")

    # What the network decided: "Cat" or "Dog".
    label = models.CharField(max_length=20, blank=True)

    # How sure the network is, from 0.0 (unsure) to 1.0 (very sure).
    # This is the number that comes out of the OUTPUT NEURON.
    confidence = models.FloatField(default=0.0)

    # When this prediction was made (filled in automatically).
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def confidence_percent(self):
        """Turn 0.87 into 87 so the page can show a friendly percentage."""
        return round(self.confidence * 100)

    def __str__(self):
        # How this row looks in the admin panel / shell.
        return f"{self.label} ({self.confidence_percent}%)"