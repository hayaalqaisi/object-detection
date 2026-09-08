"""
classifier.py
=============

WHAT IS THIS FILE FOR?
----------------------
This is where the network actually MAKES A DECISION about a new photo.

network.py built the brain. train.py (optionally) taught it. This file does
the everyday job: take one image, run it through the network, and report back
whether it looks like a "Cat" or a "Dog", plus how sure the network is.

The website calls one function from this file:

    label, confidence = classify_image("path/to/photo.jpg")

Everything else in this file exists to support that single function.


THE ONE BIG IDEA: FORWARD PROPAGATION
-------------------------------------
Making a guess is just FORWARD PROPAGATION (a "forward pass"): the image goes
in the input layer, flows through every hidden layer, and a number comes out
of the output layer. NOTHING is learned here -- the weights were already fixed
during training. It is pure calculation, from input to answer. In Keras this
one forward pass is done by:  model.predict(image)


TWO POSSIBLE NETWORKS (both only do a forward pass)
---------------------------------------------------
1. YOUR network, if you trained one (train.py saved cat_dog_model.h5).
2. A ready-made network (MobileNetV2) that we fall back to so the app works
   on day one, before anyone has trained anything.

We read their outputs slightly differently, and the comments below explain
exactly why. If you have not trained your own network yet, you can safely
ignore the custom-network path and focus on the MobileNetV2 path.
"""

import os
import numpy as np

# NOTE: we do NOT import TensorFlow at the top of this file. TensorFlow is
# slow to load, and Django runs this file every time the server starts. By
# importing TensorFlow *inside* the functions below, the website starts
# instantly and only pays the loading cost the first time someone uploads an
# image. This is a small trick for speed, not a neural-network idea.


# --------------------------------------------------------------------------
# REMEMBERING THE NETWORK BETWEEN REQUESTS
# --------------------------------------------------------------------------
# Loading a network from disk is SLOW, but using it is FAST. So we load it
# ONCE and keep it in these module-level variables. Every later prediction
# reuses the already-loaded network instead of loading it again.
#
#   _model      -> the loaded network itself
#   _model_kind -> a label telling us WHICH kind we loaded ("custom" or
#                  "pretrained"), so we know how to read its output.
# (The leading underscore is just a Python convention meaning "internal --
#  other files should not touch this directly".)
_model = None
_model_kind = None

# The place train.py would have saved a student-trained network.
CUSTOM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "cat_dog_model.h5")


def _load_model():
    """Load the network the first time we need it, then remember it."""
    global _model, _model_kind

    # If we already loaded a network earlier, do nothing and return.
    if _model is not None:
        return

    if os.path.exists(CUSTOM_MODEL_PATH):
        # OPTION A: the network YOU built and trained with train.py.
        from tensorflow.keras.models import load_model
        _model = load_model(CUSTOM_MODEL_PATH)
        _model_kind = "custom"
    else:
        # OPTION B: a ready-made network trained by someone else. It already
        # knows 1000 kinds of object, cats and dogs among them. We are still
        # only doing a forward pass through it.
        from tensorflow.keras.applications import MobileNetV2
        _model = MobileNetV2(weights="imagenet")
        _model_kind = "pretrained"


def classify_image(image_path):
    """
    The ONE function the rest of the app calls.

    Input : image_path -> where the uploaded photo is on disk.
    Output: (label, confidence)
              label      -> "Cat", "Dog", or "Unknown"
              confidence -> how sure we are, from 0.0 to 1.0
    """
    # Make sure a network is loaded and ready.
    _load_model()

    # Send the work to the correct helper, depending on which network we have.
    if _model_kind == "custom":
        return _predict_with_custom_network(image_path)
    return _predict_with_pretrained_network(image_path)


# ==========================================================================
# PATH A -- forward pass through YOUR small network (built in network.py)
# ==========================================================================
def _predict_with_custom_network(image_path):
    # We only need this Keras helper inside this function, so we import it here.
    from tensorflow.keras.preprocessing import image as keras_image

    # STEP 1 -- INPUT LAYER: load the photo at the size the network expects
    # (128x128, the same size train.py used).
    photo = keras_image.load_img(image_path, target_size=(128, 128))

    # Turn the photo into numbers, then divide by 255 so every pixel is in the
    # 0..1 range -- exactly the same rescaling train.py did while learning.
    pixels = keras_image.img_to_array(photo) / 255.0

    # The network expects a BATCH of images, not a single one. We wrap our one
    # image in a batch of size 1. Shape goes from (128,128,3) to (1,128,128,3).
    image_batch = np.expand_dims(pixels, axis=0)

    # STEP 2 -- FORWARD PASS: push the pixels through every layer.
    # The single output neuron uses Sigmoid, so the number that comes out is
    # already a probability between 0 and 1. Here it means P(dog).
    # predict() returns a batch of results; we take the first ([0]) and the
    # single output value ([0]).
    prob_dog = float(_model.predict(image_batch, verbose=0)[0][0])

    # STEP 3 -- READ THE OUTPUT NEURON.
    #   0.5 or more  -> more likely a dog; our confidence IS P(dog).
    #   below 0.5    -> more likely a cat; the confidence for "cat" is the
    #                   leftover probability, 1 - P(dog).
    if prob_dog >= 0.5:
        return "Dog", prob_dog
    return "Cat", 1.0 - prob_dog


# ==========================================================================
# PATH B -- forward pass through the ready-made MobileNetV2 network
# ==========================================================================
# MobileNetV2 can recognise 1000 different things, not just cats and dogs.
# Its output layer has 1000 neurons; these ranges tell us WHICH of those
# neurons stand for dogs and which stand for cats.
DOG_NEURONS = range(151, 269)   # positions 151..268 are dog breeds
CAT_NEURONS = range(281, 286)   # positions 281..285 are domestic cats


def _predict_with_pretrained_network(image_path):
    from tensorflow.keras.preprocessing import image as keras_image
    # MobileNetV2 was trained with a very specific way of preparing pixels, so
    # we must prepare our image the same way using its own preprocess_input.
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    # STEP 1 -- INPUT LAYER: MobileNetV2 expects 224x224 images.
    photo = keras_image.load_img(image_path, target_size=(224, 224))
    pixels = keras_image.img_to_array(photo)
    image_batch = np.expand_dims(pixels, axis=0)
    image_batch = preprocess_input(image_batch)

    # STEP 2 -- FORWARD PASS -> 1000 probabilities from a Softmax output layer.
    # Softmax makes all 1000 numbers add up to 1 (like sharing out 100% of the
    # confidence), so each number answers "how likely is it THIS object?".
    predictions = _model.predict(image_batch, verbose=0)[0]

    # STEP 3 -- FIND THE BEST CAT GUESS AND THE BEST DOG GUESS.
    # Careful: there are ~118 dog breeds but only ~5 cat types in the list. If
    # we ADDED UP each group's probabilities, the many dog classes would almost
    # always win -- even for an obvious cat. So instead we take the SINGLE
    # strongest cat neuron and the SINGLE strongest dog neuron and compare
    # just those two. That is a fair, size-independent comparison.
    best_cat_neuron = max(CAT_NEURONS, key=lambda i: predictions[i])
    best_dog_neuron = max(DOG_NEURONS, key=lambda i: predictions[i])
    cat_score = float(predictions[best_cat_neuron])
    dog_score = float(predictions[best_dog_neuron])

    # STEP 4 -- TURN THE TWO SCORES INTO A "CAT vs DOG ONLY" CONFIDENCE.
    # We divide by the total of the two so the confidence is relative to just
    # cat-vs-dog, ignoring the other 995 objects the network knows about.
    total = cat_score + dog_score
    if total == 0:
        # The network saw neither a cat nor a dog with any confidence.
        return "Unknown", 0.0

    if dog_score > cat_score:
        return "Dog", dog_score / total
    return "Cat", cat_score / total