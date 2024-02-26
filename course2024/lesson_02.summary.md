# Lesson 2

## The State of Deep Learning

### Computer Vision

- Deep learning has **not yet been applied** to image analysis in every domain, but where it has, it often **surpasses human-level performance** in object recognition.
- Object recognition enables computers to **identify items in images as well or better than humans**, including experts like radiologists.
- Deep learning excels in **object detection**, identifying and highlighting object locations and names within images.
- It also performs well in **segmentation**, where each pixel is categorized based on the object it belongs to.
- Deep learning models struggle with images that **differ significantly in structure or style** from their training data, such as black-and-white or hand-drawn images if these were not included in the training set (**out-of-domain data**).
- **Image labeling**, essential for object detection systems, is slow and costly, driving research into making this process faster and requiring fewer labels.
- **Data augmentation**, such as rotating images or adjusting brightness, is an effective strategy for enhancing model training and is applicable to various data types.
- Problems not traditionally seen as computer vision issues may benefit from being approached as such, exemplified by **converting sounds into visual waveforms** for image-based analysis.
- **Integration of Text and Images**: Models can be trained with images as input and English captions as output, learning to autonomously produce fitting captions for new images. However, there's no guarantee the generated captions will always be correct. Therefore, deep learning is advised to be used in collaboration with human oversight rather than as a fully automated solution, enhancing productivity and accuracy.

### NLP

- **Machine Translation**: Deep learning models, especially sequence-to-sequence (seq2seq) and Transformer models.
- **Sentiment Analysis**: from customer reviews to social media posts, aiding businesses in understanding consumer attitudes and preferences.
- **Chatbots and Virtual Assistants**: responsive, context-aware chatbots and virtual assistants, capable of handling complex, nuanced interactions with users in a conversational manner.
- **Text Summarization**: allowing for the extraction of key points or the generation of concise summaries from large documents, making information retrieval more efficient.
- **Named Entity Recognition (NER)**: identifying and classifying named entities in text, crucial for information extraction, content classification, and data analysis applications.
- **Speech Recognition**: highly accurate speech recognition systems used from voice-activated commands to transcription services.
- **Natural Language Generation (NLG)**: enabling the generation of coherent, contextually relevant text.
- **Question Answering Systems**: comprehend and provide accurate answers to queries based on vast amounts of structured and unstructured data.
- **Content Moderation**: detecting and filtering inappropriate or harmful content in online platforms, ensuring safer digital environments for users.

### Tabular Data

- **Ensemble Approach**: It's common to use deep learning alongside other models, such as random forests or gradient boosting machines, rather than replacing them entirely.
- **Incremental Improvements**: Adding deep learning to existing systems may not lead to dramatic improvements if those systems already use effective tabular modeling tools.
- **Enhanced Data Handling**: increases the types of data that can be included in models, such as natural language fields (e.g., book titles, reviews) and high-cardinality categorical data (e.g., zip codes, product IDs).
- **Training Time Consideration**: Deep learning models typically require more training time compared to random forests or gradient boosting machines, although advancements like GPU acceleration with libraries such as [RAPIDS](https://rapids.ai/) are reducing these times.

### Recommendation System

- **Nature of Recommendation Systems**: Recommendation systems are viewed as a specific application of tabular data, often dealing with high-cardinality categorical variables for users and products.
- **Data Representation**: Companies like Amazon create a sparse matrix from customer purchases, with customers and products represented by rows and columns, respectively.
- **Collaborative Filtering**: This technique is used to predict customer preferences by analyzing purchase patterns, filling in the matrix to recommend products based on similarities among users.
- **Deep Learning Advantage**: proficiency with high-cardinality categorical data, and its capability is enhanced when combined with diverse data types like natural language or images.
- **Comprehensive Data Integration**: e.g. including user metadata and transaction history into the recommendation process.
- **Limitations of Machine Learning in Recommendations**: recommendations might not always be useful, even if they align with user preferences.

## Data Augmentation

Data augmentation is a strategy used in machine learning to increase the diversity and amount of training data without actually collecting new data. This is done by creating modified versions of the existing data. For example, in image data, common augmentation techniques include:

- Rotation: Rotating the image by a certain angle.
- Translation: Shifting the image along the X or Y direction.
- Scaling: Increasing or decreasing the size of the image.
- Flipping: Flipping the image horizontally or vertically.
- Noise Injection: Adding random noise to the image.
- Brightness and Contrast Adjustment: Changing the brightness or contrast of the image.

## Data Challenges

- `Out-of-Domain Data`: That is to say, there may be data that our model sees in production which is very different to what it saw during training. There isn't really a complete technical solution to this problem; instead, we have to be careful about our approach to rolling out the technology.

- `Domain Shift`: This is a related problem where the distribution of the data changes over time. For example, a model might be trained on a certain set of data, but then the data it's asked to make predictions on slowly changes or drifts away from the original training data. This could happen due to changes in user behavior, changes in the environment, or other factors.


## Starting a Project

- end-to-end iteration approach
- get started where you alreaday have data

## Deploying a Project

1. Manual Process (humans check all predictions)
2. Limited scope Deployment (time or geographically limited)
3. Gradual Expansion

## Practical Tips

- Train the model before cleaning the data.
- Run the Confusion Matrix to compare predicted with actual (if your labels are categories). This will convey which categories are difficult to identify.
- Print top losses, and check for wrongly predicted items with high confidence. **In practice, you can have a bad loss either by being wrong and confident, or by being right and unconfident.**
- Clean the items that are wrongly classified.

### Jupyter Notebook Help Functions

```
??verify_images
```
would show:
```
Signature: verify_images(fns)
Source:   
def verify_images(fns):
    "Find images in `fns` that can't be opened"
    return L(fns[i] for i,o in
             enumerate(parallel(verify_image, fns)) if not o)
File:      ~/git/fastai/fastai/vision/utils.py
Type:      function
```
This tells us what argument the function accepts (`fns`), then shows us the source code and the file it comes from. Looking at that source code, we can see it applies the function `verify_image` in parallel and only keeps the image files for which the result of that function is `False`, which is consistent with the doc string: it finds the images in `fns` that can't be opened.

Here are some other features that are very useful in Jupyter notebooks:

- At any point, if you don't remember the exact spelling of a function or argument name, you can press Tab to get autocompletion suggestions.
- When inside the parentheses of a function, pressing Shift and Tab simultaneously will display a window with the signature of the function and a short description. Pressing these keys twice will expand the documentation, and pressing them three times will open a full window with the same information at the bottom of your screen.
- In a cell, typing `?func_name` and executing will open a window with the signature of the function and a short description.
- In a cell, typing `??func_name` and executing will open a window with the signature of the function, a short description, and the source code.
- If you are using the fastai library, we added a `doc` function for you: executing `doc(func_name)` in a cell will open a window with the signature of the function, a short description and links to the source code on GitHub and the full documentation of the function in the [library docs](https://docs.fast.ai).
- Unrelated to the documentation but still very useful: to get help at any point if you get an error, type `%debug` in the next cell and execute to open the [Python debugger](https://docs.python.org/3/library/pdb.html), which will let you inspect the content of every variable.

## References

- [Jupyter Notebooks Extensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html) for Table of Contents, Collapsible Headings
- [Gradio + HuggingFace Spaces: A Tutorial](https://www.tanishq.ai/blog/posts/2021-11-16-gradio-huggingface.html)
- [fastai/fastsetup](https://github.com/fastai/fastsetup)
- [mamba](https://github.com/mamba-org/mamba)
- [nvdev](https://nbdev.fast.ai/)

## Books

- [Designing Great Data Products](https://www.oreilly.com/radar/drivetrain-approach-data-products/)
- [Building Machine Learning Powered Applications](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/)
