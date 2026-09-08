"""
network.py
==========

WHAT IS THIS FILE FOR?
----------------------
This file has ONE job: to BUILD the neural network that will learn to tell
cats from dogs.

Think of it as drawing the blueprint of a brain. This file does NOT:
  * teach the brain anything   -> that happens in train.py
  * use the brain to guess     -> that happens in classifier.py

It only describes the SHAPE of the network: which layers it has, in what
order, and how big each one is.

WHY IS "BUILDING" IN ITS OWN FILE?
----------------------------------
Because two different files need the exact same network shape:
  * train.py builds a brand-new (empty) network and teaches it.
  * (Optionally) we could rebuild the same shape elsewhere for testing.
Keeping the blueprint in one place means there is a single, clear answer to
the question "what does our network look like?".


----------------------------------
A neural network is made of LAYERS. Each layer is a group of tiny units
called NEURONS. Information flows in one direction, layer by layer:

    the image  ->  layer 1  ->  layer 2  ->  ...  ->  final answer

  * INPUT LAYER   : the raw image (every pixel is a number the network sees)
  * HIDDEN LAYERS : the middle layers that gradually find patterns
                    (edges -> shapes -> parts like "an ear" or "an eye")
  * OUTPUT LAYER  : the final neuron(s) that give the answer

Pushing data forward through the layers to get an answer is called
FORWARD PROPAGATION. Building the network here does not do any of that yet.
"""

# We import two helpers from Keras (which lives inside TensorFlow):
#   * "layers" gives us ready-made layer types (Conv2D, Dense, ...). We do not
#     have to program a neuron by hand; Keras already knows how one works.
#   * "models" lets us stack those layers together into one whole network.
from tensorflow.keras import layers, models


# We wrap the network in a FUNCTION so that anyone who needs a fresh, empty
# network can just call build_cnn() and get one. A function is reusable: call
# it once in train.py, call it again later, and you always get the same shape.
#
# input_shape says what ONE image looks like to the network:
#   (128, 128, 3)  ->  128 pixels tall, 128 pixels wide, 3 colour channels
#                      (the 3 is Red, Green, Blue).
def build_cnn(input_shape=(128, 128, 3)):
    """Create and return a brand-new (untrained) cat-vs-dog network."""

    # models.Sequential means "a stack of layers, one after another".
    # The data enters the first layer, flows straight down the list, and the
    # last layer produces the answer. This is the simplest way to describe a
    # network.
    model = models.Sequential([

        # ------------------------------------------------------------------
        # INPUT LAYER
        # ------------------------------------------------------------------
        # This just tells the network what size of image to expect. No maths
        # happens here; it is the "doorway" the pixels come through.
        layers.Input(shape=input_shape),

        # ------------------------------------------------------------------
        # HIDDEN LAYER 1  ->  find the simplest patterns (edges, corners)
        # ------------------------------------------------------------------
        # A Conv2D layer slides small "magnifying glasses" (called filters)
        # across the image, looking for a pattern in each little patch.
        #   * 16      = how many different patterns this layer looks for
        #   * (3, 3)  = each magnifying glass covers a 3x3 patch of pixels
        #   * relu    = the activation (explained at the bottom of this file):
        #               it keeps positive signals and zeroes out negatives.
        layers.Conv2D(16, (3, 3), activation="relu"),

        # MaxPooling2D shrinks the image by keeping only the STRONGEST signal
        # in each little area. This throws away detail we do not need and
        # makes the network faster and less fussy about exact positions.
        layers.MaxPooling2D(),

        # ------------------------------------------------------------------
        # HIDDEN LAYER 2  ->  combine edges into shapes/textures
        # ------------------------------------------------------------------
        # We use MORE filters (32) than before. Deeper layers usually look for
        # more, and more complicated, patterns than the layers before them.
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        # ------------------------------------------------------------------
        # HIDDEN LAYER 3  ->  combine shapes into "parts" (an ear, an eye)
        # ------------------------------------------------------------------
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        # ------------------------------------------------------------------
        # FLATTEN  ->  turn the 2D picture into one long line of numbers
        # ------------------------------------------------------------------
        # The layers above work on a 2D grid (like a picture). The next layers
        # (Dense) work on a plain 1D list. Flatten unrolls the grid into that
        # single list so the two styles of layer can connect.
        layers.Flatten(),

        # ------------------------------------------------------------------
        # A NORMAL HIDDEN LAYER of 64 neurons ("Dense" = fully connected)
        # ------------------------------------------------------------------
        # "Dense" means every neuron here is connected to every number coming
        # in. This layer mixes all the patterns found above and decides what
        # is important for the final cat/dog decision.
        layers.Dense(64, activation="relu"),

        # ------------------------------------------------------------------
        # OUTPUT LAYER  ->  ONE neuron with the Sigmoid activation
        # ------------------------------------------------------------------
        # We use ONE output neuron because this is a yes/no question:
        # "is it a dog?". Sigmoid squashes the neuron's number into the range
        # 0..1, so we can read it as a PROBABILITY that the image is a DOG:
        #     close to 1  ->  confident DOG
        #     close to 0  ->  confident CAT
        #     around 0.5  ->  unsure
        #
        # LABEL ORDER (why cats are 0 and dogs are 1):
        # In train.py the folders are read in ALPHABETICAL order, so
        #     "cats"  ->  0   and   "dogs"  ->  1.
        # classifier.py then reads this neuron as "0.5 or more means Dog",
        # which matches. Keep the folders named "cats" and "dogs".
        layers.Dense(1, activation="sigmoid"),
    ])

    # We hand the finished (but still untrained) network back to whoever
    # called this function.
    return model


# ==========================================================================
# ABOUT THE TWO ACTIVATION FUNCTIONS WE USED
# ==========================================================================
# After a neuron adds up its inputs, it passes the total through an
# "activation function" before sending it onward. The activation decides how
# the neuron reacts. We used two:
#
#   ReLU  (in the HIDDEN layers)
#   ---------------------------------------------------------------
#   Rule:  negative number -> 0 ,  positive number -> unchanged.
#   In one line:  relu(x) = max(0, x)
#   Examples:     relu(-3) = 0 ,  relu(2.5) = 2.5
#   Why: it is fast, and it lets the network learn "bendy", non-straight-line
#   patterns instead of only straight lines.
#
#   Sigmoid  (in the OUTPUT neuron)
#   ---------------------------------------------------------------
#   Rule:  squashes ANY number smoothly into the range 0..1.
#   In one line:  sigmoid(x) = 1 / (1 + e^(-x))
#   Examples:     sigmoid(-5) ~ 0.01 ,  sigmoid(0) = 0.5 ,  sigmoid(5) ~ 0.99
#   Why: because the result is between 0 and 1, we can read it as a
#   probability, which is perfect for a yes/no answer.


# This runs ONLY if you execute "python network.py" directly. It prints a
# neat table of the layers so you can see the network you just built. It is a
# handy way to check your work without training anything.
if __name__ == "__main__":
    model = build_cnn()
    model.summary()