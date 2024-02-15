# Practical Deep Learning @ fast.ai

Learning notes and projects built for the Deep Learning course by [fast.ai](https://course.fast.ai/).

01. Bird Classifier

[[Notebook]](https://github.com/dcarpintero/fastai-deeplearning/blob/main/course2024/lesson_01.ipynb) 
[[Lesson Summary]](https://github.com/dcarpintero/fastai-deeplearning/blob/main/course2024/lesson_01.summary.md)

02. Interstellar (Astronomical Classifier)

[[Open in HuggingFace]](https://huggingface.co/spaces/dcarpintero/interstellar) 
[[Model]](https://huggingface.co/dcarpintero/fastai-interstellar-class)
[[Notebook]](https://github.com/dcarpintero/fastai-deeplearning/blob/main/course2024/lesson_02.ipynb) 
[[Lesson Summary]](https://github.com/dcarpintero/fastai-deeplearning/blob/main/course2024/lesson_02.summary.md)

`[deep-learning]` `[data-augmentation]` `[ResNet-50]` `[transfer-learning]`

Classify images of astronomical objects such as galaxies, nebulae, comets, asteroids, quasars, and star clusters. Built by:
- creating a custom dataset (less than 150 images per label) using Bing search API;
- augmenting the dataset; and,
- fine tuning ResNet50 (1 + 3 epochs) in paperspace.com.

It reaches 84% accuracy on class level:

<p align="center">
  <img src="./course2024/static/hg.01.png">
</p>

and 94.1% accuracy at object level (on the validation set):

<p align="center">
  <img src="./course2024/static/hg.02.png">
</p>
