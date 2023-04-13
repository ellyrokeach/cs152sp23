import torch
import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
classifier("I love this!")

demo = gr.Interface.load(
             "huggingface/michellejieli/emotion_text_classifier")

demo.launch()