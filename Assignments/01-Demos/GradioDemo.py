"""
Many examples are taken from the Hugging Face documentation and the Gradio Demos.
- https://huggingface.co/docs
- https://www.gradio.app/demos/
"""

from argparse import ArgumentParser

from random import randint

import gradio as gr
import numpy as np
import torch

from diffusers import DiffusionPipeline

from transformers import (
    pipeline,
    MaskFormerFeatureExtractor,
    MaskFormerForInstanceSegmentation,
)


def get_cartoonify():
    """Encapsulate AnimeGAN functionality."""
    animegan = torch.hub.load(
        "AK391/animegan2-pytorch:main",
        "generator",
        pretrained=True,
        progress=False,
    )
    face2paint = torch.hub.load(
        "AK391/animegan2-pytorch:main", "face2paint", size=512, side_by_side=False
    )

    def cartoonify(image):
        return face2paint(animegan, image)

    return cartoonify


def visualize_instance_seg_mask(mask):
    image = np.zeros((mask.shape[0], mask.shape[1], 3))
    labels = np.unique(mask)
    label2color = {
        label: (randint(0, 1), randint(0, 255), randint(0, 255)) for label in labels
    }
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j, :] = label2color[mask[i, j]]
    image = image / 255
    return image


def get_segmenter():
    """Encapsulate image segmentation functionality."""
    segmenter_model = MaskFormerForInstanceSegmentation.from_pretrained(
        "facebook/maskformer-swin-tiny-ade"
    ).to(torch.device("cpu"))
    segmenter_model.eval()

    preprocessor = MaskFormerFeatureExtractor.from_pretrained(
        "facebook/maskformer-swin-tiny-ade"
    )

    def segmenter(image):
        target_size = (image.shape[0], image.shape[1])
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = segmenter_model(**inputs)
        outputs.class_queries_logits = outputs.class_queries_logits.cpu()
        outputs.masks_queries_logits = outputs.masks_queries_logits.cpu()
        results = (
            preprocessor.post_process_segmentation(
                outputs=outputs, target_size=target_size
            )[0]
            .cpu()
            .detach()
        )
        results = torch.argmax(results, dim=0).numpy()
        results = visualize_instance_seg_mask(results)
        return results

    return segmenter


def get_diffuser():
    image_generator = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )
    image_generator.to("cuda")

    def generator(prompt):
        return image_generator(prompt).images[0]

    return generator


def get_captioner():
    image_captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")

    def caption_image(image):
        return image_captioner(image)[0]["generated_text"]

    return caption_image


demos = {
    "Text Classifier": gr.Interface.from_pipeline(
        pipeline(task="sentiment-analysis"),
        title="Sentiment Analysis",
        description="Type some text and this model will classify it as positive or negative in sentiment.",
    ),
    "Text Generator": gr.Interface.from_pipeline(
        pipeline(task="text-generation"),
        title="Text Generation",
        description="Type some starter text and a model will fill in the rest.",
    ),
    "Text Filler": gr.Interface.from_pipeline(
        pipeline(task="fill-mask"),
        title="Fill In the Missing Mask",
        description="Type a sentence and replace a word with `<mask>`",
    ),
    "English to French": gr.Interface.from_pipeline(
        pipeline(task="translation_en_to_fr"),
        title="Translation",
        description="Translate an English sentence into French.",
    ),
    "Question/Answer": gr.Interface.from_pipeline(
        pipeline(model="deepset/roberta-base-squad2"),
        title="Ask a Question, Receive an Answer",
        description="Write some that includes an answer to a question. Then, ask a question.",
    ),
    "Image Classifier": gr.Interface.from_pipeline(
        pipeline(task="image-classification"),
        title="Image Classification",
        description="Classify the main entity in an image.",
    ),
    "Image Segmenter": gr.Interface(
        fn=get_segmenter(),
        inputs="image",
        outputs="image",
        title="Image Segmenter",
        description="Upload an image the model will segment pixels into different entities.",
    ),
    "Image Captioner": gr.Interface(
        fn=get_captioner(),
        inputs=gr.inputs.Image(type="pil"),
        outputs="label",
        title="Caption the Provided Image",
        description="The model will provide a caption for the given image.",
    ),
    "Image to Anime": gr.Interface(
        fn=get_cartoonify(),
        inputs=gr.inputs.Image(type="pil"),
        outputs=gr.outputs.Image(type="pil"),
        title="Cartoonify An Image",
        description="Gradio Demo for AnimeGanv2 Face Portrait. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below. Please use a cropped portrait picture for best results similar to the examples below.",
    ),
    "Image Generator": gr.Interface(
        fn=get_diffuser(),
        inputs=gr.Text(),
        outputs=gr.outputs.Image(type="pil"),
        title="Image Generation",
        description="Give a phrase and let the model generate an image.",
    ),
    "Text-to-Speech": gr.Interface.load(
        "huggingface/facebook/fastspeech2-en-ljspeech",
        title="Audio Generator",
        description="Give me something to say!",
    ),
}

demo = gr.TabbedInterface(
    list(demos.values()), list(demos.keys()), title="CS 152 Neural Network Demos"
)

if __name__ == "__main__":
    argparser = ArgumentParser("CS 152 Demo Assignment")
    argparser.add_argument("port", type=int, help="Your personal forwarded port.")
    args = argparser.parse_args()
    demo.launch(server_port=args.port)
