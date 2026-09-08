"""
train.py   (this is the "LEARNING" part -- optional, but the heart of ML)
========================================================================

WHAT IS THIS FILE FOR?
----------------------
network.py built an EMPTY brain. It has layers, but it knows nothing yet -- if
you asked it "cat or dog?" right now it would basically guess randomly.

This file TEACHES that brain, by showing it thousands of example photos that
are already labelled "cat" or "dog". After enough practice, the network's
internal numbers (its "weights") settle on values that make good guesses.
That process of practising and improving is called TRAINING.

When training finishes, we SAVE the trained brain to a file called
cat_dog_model.h5. The website (classifier.py) then loads that file and uses
YOUR network instead of the ready-made one.

WHERE DOES THE DATA COME FROM?
------------------------------
You provide it, in this exact folder layout:

    object_detection/ml/data/
      train/                <- photos used for LEARNING
        cats/   (lots of cat photos)
        dogs/   (lots of dog photos)
      validation/           <- photos used only for CHECKING progress
        cats/
        dogs/

(You can download a free "cats vs dogs" image set online.)

HOW TO RUN IT
-------------
From the project root, run:

    python object_detection/ml/train.py

When it finishes it saves cat_dog_model.h5 next to this file.
"""

import os

# ImageDataGenerator is a Keras helper that reads image files from folders and
# feeds them to the network in small groups. We use it because it saves us
# from writing our own "open every file and turn it into numbers" code.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# build_cnn is the network blueprint we wrote in network.py. Because we run
# THIS file directly, Python automatically looks in this file's own folder,
# and network.py sits right next to it -- so this plain import works.
from network import build_cnn


# --------------------------------------------------------------------------
# SETTINGS
# --------------------------------------------------------------------------
# We put these numbers at the top, with names, so they are easy to find and
# change later (this is nicer than scattering magic numbers through the code).

# Every image is resized to this size before the network sees it. It MUST
# match the input_shape used in network.py (128 x 128).
IMG_SIZE = (128, 128)

# BATCH SIZE = how many images the network looks at before it updates itself.
# We do NOT show it all images at once -- that would need too much memory and
# learn more slowly. Instead we work in small "batches" of 32.
BATCH_SIZE = 32

# How many times we go through the WHOLE training set. One full pass over
# every training image is called one EPOCH. More epochs = more practice
# (but too many can cause the network to just memorise the examples).
EPOCHS = 5

# Work out the folder paths relative to THIS file, so the script works no
# matter which folder you run it from.
HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")
SAVE_PATH = os.path.join(HERE, "cat_dog_model.h5")


def main():
    # ----------------------------------------------------------------------
    # STEP 1 -- PREPARE THE DATA
    # ----------------------------------------------------------------------
    # Pixels normally range from 0 to 255. Neural networks learn better when
    # the numbers are small, so we rescale every pixel to the range 0..1 by
    # dividing by 255. That is what "rescale=1.0 / 255" does.
    data_loader = ImageDataGenerator(rescale=1.0 / 255)

    # flow_from_directory looks inside a folder, finds the sub-folders
    # ("cats", "dogs"), and uses those folder NAMES as the labels. It then
    # streams the images to us in batches, already resized and rescaled.
    #
    #   class_mode="binary"  -> there are exactly two classes, so each image
    #                           gets a single 0 or 1 label (cats=0, dogs=1,
    #                           because folders are read alphabetically).
    training_images = data_loader.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    # The VALIDATION images are kept separate. The network never learns from
    # them; we only use them to CHECK how well it does on photos it has not
    # practised on. That tells us if it is truly learning or just memorising.
    validation_images = data_loader.flow_from_directory(
        os.path.join(DATA_DIR, "validation"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    # ----------------------------------------------------------------------
    # STEP 2 -- CREATE THE (EMPTY) NETWORK
    # ----------------------------------------------------------------------
    # We ask network.py for a fresh, untrained network of the right shape.
    model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # ----------------------------------------------------------------------
    # STEP 3 -- TELL THE NETWORK *HOW* TO LEARN
    # ----------------------------------------------------------------------
    # compile() attaches three tools to the network before training:
    #
    #   loss = "binary_crossentropy"
    #     The LOSS is a single number that measures HOW WRONG a guess was.
    #     Big loss = very wrong, small loss = nearly right. The whole point of
    #     training is to make this number as small as possible.
    #     "binary_crossentropy" is the standard loss for a yes/no question.
    #
    #   optimizer = "adam"
    #     The OPTIMIZER is the rule that actually adjusts the network's weights
    #     after each batch so that next time the loss is a bit smaller. In
    #     other words, it is what makes the network LEARN from its mistakes.
    #     "adam" is a popular, reliable optimizer -- a smart version of the
    #     simple learning rule you saw in the lesson.
    #
    #   metrics = ["accuracy"]
    #     This is just for US to read. Accuracy = the % of images guessed
    #     correctly. It does not change how the network learns; it only lets
    #     us watch progress in a way that is easy to understand.
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # ----------------------------------------------------------------------
    # STEP 4 -- TRAIN (this is where the learning happens)
    # ----------------------------------------------------------------------
    # fit() runs the whole training loop for us. For each batch of images it:
    #   1. FORWARD PASS  : sends the images through the network to get guesses.
    #   2. MEASURE LOSS  : compares those guesses to the true labels.
    #   3. BACKPROPAGATION: works out which weights caused the mistakes.
    #   4. UPDATE WEIGHTS: the optimizer nudges those weights to do better.
    # It repeats this for every batch, and repeats the whole thing EPOCHS times.
    #
    # After each epoch it also checks the validation images so you can see, in
    # the printout, whether accuracy is going up (good) or getting stuck.
    model.fit(
        training_images,
        validation_data=validation_images,
        epochs=EPOCHS,
    )

    # ----------------------------------------------------------------------
    # STEP 5 -- SAVE THE TRAINED NETWORK
    # ----------------------------------------------------------------------
    # Training can take a while, so we save the result to a file. Saving keeps
    # everything the network LEARNED (its weights) so we never have to train
    # again -- the website can just load this file and start guessing.
    model.save(SAVE_PATH)
    print(f"\nDone! Saved your trained network to:\n  {SAVE_PATH}")


# This "if" makes sure main() only runs when you launch the file yourself
# (python train.py), not if some other file imports it.
if __name__ == "__main__":
    main()