import torch
import gradio as gr
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Sentiment analysis model
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
classifier("I love this!")

demo1 = gr.Interface.load(
             "huggingface/michellejieli/emotion_text_classifier")
demo1.launch()

# 2. Translation model 
tokenizer = AutoTokenizer.from_pretrained("salesken/translation-spanish-and-portuguese-to-english")
model = AutoModelForSeq2SeqLM.from_pretrained("salesken/translation-spanish-and-portuguese-to-english")
snippet = "Eu estou falando em L�ngua Portuguesa."
inputs = tokenizer.encode(
    snippet, return_tensors="pt",padding=True,max_length=512,truncation=True)
outputs = model.generate(
    inputs, max_length=128, num_beams=None, early_stopping=True)
translated = tokenizer.decode(outputs[0]).replace('<pad>',"").strip().lower()
print(translated)

demo2 = gr.Interface.load(
    "huggingface/salesken/translation-spanish-and-portuguese-to-english")
demo2.launch()