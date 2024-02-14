import gradio as gr
from fastai.vision.all import *
import skimage

model = load_learner("model.pkl")


def predict(img):
    labels = model.dls.vocab
    img = PILImage.create(img)
    pred, pred_idx, probs = model.predict(img)
    return dict(map(labels, map(float, probs)))


title = "Interstellar Classifier"
description = "Built for fast.ai 'Practical Deep Learning'"
examples = [
    "m31_andromeda.jpg",
    "m51_whirlpool.jpg",
    "m104_sombrero.jpg",
    "m42_orion.jpg",
    "m82_cigar.jpg",
    "ngc_1300.jpg",
]
enable_queue = True

gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(shape=(512, 512)),
    outputs=gr.outputs.Label(num_top_classes=6),
    title=title,
    description=description,
    examples=examples,
    enable_queue=enable_queue,
).launch()
