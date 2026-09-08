from django.shortcuts import redirect, render

from object_detection.ml.classifier import classify_image
from object_detection.models import Prediction


def upload_view(request):
    # Did the user submit the form WITH an image?
    if request.method == "POST" and request.FILES.get("image"):
        # 1. Save the uploaded image into a new database row.
        # We save first so the file exists on disk for the network to read.
        prediction = Prediction(image=request.FILES["image"])
        prediction.save()

        # 2. Forward propagation: run the image through the neural network.
        label, confidence = classify_image(prediction.image.path)

        # 3. Store what the network decided.
        prediction.label = label
        prediction.confidence = confidence
        prediction.save()

        # 4. Show the user the results obtained.
        return redirect("result", pk=prediction.pk)

    # If the user did not submit the form, simply show the upload page.
    return render(
        request,
        template_name="object_detection/upload.html",
    )