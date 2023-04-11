#!/usr/bin/env python

"""
Run this file with:
GRADIO_SERVER_PORT=<port> python FrankOrFraryGradioApp.py </path/to/model>

Where
- <port> is your server port number
- </path/to/model> is the path to the model you want to use for inference
"""

import gradio as gr
from fastai.vision.all import *

# Load the trained model
path = Path(sys.argv[1])
model = load_learner(path)


def classify(img):
    prediction = model.predict(img)
    label, label_index, probabilities = prediction
    label_prob = probabilities[label_index].item()

    return {
        "Frank": label_prob if label == "Frank" else 1 - label_prob,
        "Frary": label_prob if label == "Frary" else 1 - label_prob,
    }


title = "Frank or Frary? I'll Decide!"
website = "A demo for [CS 152](https://cs.pomona.edu/classes/cs152/)"

iface = gr.Interface(
    fn=classify,
    inputs="image",
    outputs="label",
    title=title,
    article=website,
).launch()
