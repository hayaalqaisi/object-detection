"""
Image classification helpers for comparing MobileNetV2 and EfficientNetB0.

Both models use weights pre-trained on ImageNet. Their predictions are reduced
to a Cat-versus-Dog result by comparing the strongest ImageNet cat class with
the strongest ImageNet dog class.
"""

import os
import time

import numpy as np


CUSTOM_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "cat_dog_model.h5",
)

_custom_model = None
_mobilenet_model = None
_efficientnet_model = None

# Zero-based ImageNet class indexes used by Keras.
DOG_NEURONS = range(151, 269)
CAT_NEURONS = range(281, 286)


def classify_image(image_path):
    """
    Classify an image for the existing Django website.

    If a custom model created by train.py exists, use it. Otherwise, preserve
    the existing website behavior by using MobileNetV2.

    Returns:
        tuple: (label, confidence)
    """
    if os.path.exists(CUSTOM_MODEL_PATH):
        return _predict_with_custom_model(image_path)

    return classify_with_mobilenet(image_path)


def classify_image_with_model(image_path, model_name):
    """
    Classify an image using a specifically selected pre-trained model.

    Supported model names:
        mobilenetv2
        efficientnetb0

    Returns:
        tuple: (label, confidence)
    """
    normalized_name = model_name.strip().lower()

    if normalized_name == "mobilenetv2":
        return classify_with_mobilenet(image_path)

    if normalized_name == "efficientnetb0":
        return classify_with_efficientnet(image_path)

    raise ValueError(
        "Unsupported model. Use 'mobilenetv2' or 'efficientnetb0'."
    )


def classify_image_timed(image_path, model_name):
    """
    Classify an image and measure prediction time.

    Returns:
        tuple: (label, confidence, elapsed_seconds)
    """
    start_time = time.perf_counter()

    label, confidence = classify_image_with_model(
        image_path=image_path,
        model_name=model_name,
    )

    elapsed_seconds = time.perf_counter() - start_time

    return label, confidence, elapsed_seconds


def classify_with_mobilenet(image_path):
    """
    Classify an image using MobileNetV2.

    Returns:
        tuple: (label, confidence)
    """
    global _mobilenet_model

    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    if _mobilenet_model is None:
        _mobilenet_model = MobileNetV2(weights="imagenet")

    predictions = _predict_imagenet_model(
        image_path=image_path,
        model=_mobilenet_model,
        preprocess_input=preprocess_input,
        target_size=(224, 224),
    )

    return _get_cat_or_dog(predictions)


def classify_with_efficientnet(image_path):
    """
    Classify an image using EfficientNetB0.

    Returns:
        tuple: (label, confidence)
    """
    global _efficientnet_model

    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input

    if _efficientnet_model is None:
        _efficientnet_model = EfficientNetB0(weights="imagenet")

    predictions = _predict_imagenet_model(
        image_path=image_path,
        model=_efficientnet_model,
        preprocess_input=preprocess_input,
        target_size=(224, 224),
    )

    return _get_cat_or_dog(predictions)


def _predict_imagenet_model(
    image_path,
    model,
    preprocess_input,
    target_size,
):
    """
    Prepare one image and perform forward propagation.

    Returns:
        numpy.ndarray: The model's 1,000 ImageNet probabilities.
    """
    from tensorflow.keras.preprocessing import image as keras_image

    photo = keras_image.load_img(
        image_path,
        target_size=target_size,
    )

    pixels = keras_image.img_to_array(photo)
    image_batch = np.expand_dims(pixels, axis=0)
    image_batch = preprocess_input(image_batch)

    return model.predict(image_batch, verbose=0)[0]


def _get_cat_or_dog(predictions):
    """
    Convert ImageNet probabilities into a Cat-versus-Dog result.

    Returns:
        tuple: (label, confidence)
    """
    best_cat_neuron = max(
        CAT_NEURONS,
        key=lambda index: predictions[index],
    )

    best_dog_neuron = max(
        DOG_NEURONS,
        key=lambda index: predictions[index],
    )

    cat_score = float(predictions[best_cat_neuron])
    dog_score = float(predictions[best_dog_neuron])

    total_score = cat_score + dog_score

    if total_score == 0:
        return "Unknown", 0.0

    if dog_score > cat_score:
        return "Dog", dog_score / total_score

    return "Cat", cat_score / total_score


def _predict_with_custom_model(image_path):
    """
    Classify an image using the custom binary CNN created by train.py.

    Returns:
        tuple: (label, confidence)
    """
    global _custom_model

    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image

    if _custom_model is None:
        _custom_model = load_model(CUSTOM_MODEL_PATH)

    photo = keras_image.load_img(
        image_path,
        target_size=(128, 128),
    )

    pixels = keras_image.img_to_array(photo) / 255.0
    image_batch = np.expand_dims(pixels, axis=0)

    dog_probability = float(
        _custom_model.predict(
            image_batch,
            verbose=0,
        )[0][0]
    )

    if dog_probability >= 0.5:
        return "Dog", dog_probability

    return "Cat", 1.0 - dog_probability