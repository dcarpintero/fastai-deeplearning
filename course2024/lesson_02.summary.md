# Lesson 2

## The State of Computer Vision

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

### 




## Starting a Project

- end-to-end iteration approach;
- get started where you alreaday have data;
