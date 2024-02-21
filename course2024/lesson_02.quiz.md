# Lesson2 Quiz

> 1) Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.

- clean images in training vs. blurry, motion, distant, covered bear images in prod.
- daylight images in training vs. night images in prod (and viceversa).
- the training set is biased towards a certain feature (e.g. color, type).
- unreal bear images (e.g. from comic characters) in the training dataset.

To mitigate these issues, it is useful to recognize when unexpected image types arise in production (checking for *out-of-domain* data).

> 2) Where do text models currently have a major deficiency?

- lack of ground truth and up-to-date data (partially mitigated with RAG, fine-tuning, temperature adjustment, and large context windows).
- struggle to work in multilingual mode (Cohere just released Aya on Feb 2024 https://txt.cohere.com/aya/).

> 3) What are possible negative societal implications of text generation models?

> 4) In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?

> 5) What kind of tabular data is deep learning particularly good at?

> 6) What's a key downside of directly using a deep learning model for recommendation systems?

> 7) What are the steps of the Drivetrain Approach?

> 8) How do the steps of the Drivetrain Approach map to a recommendation system?

> 9) Create an image recognition model using data you curate, and deploy it on the web.

[]().

> 10) What is DataLoaders?

> 11) What four things do we need to tell fastai to create DataLoaders?

> 12) What does the splitter parameter to DataBlock do?

> 13) How do we ensure a random split always gives the same validation set?

> 14) What letters are often used to signify the independent and dependent variables?

> 15) What's the difference between the crop, pad, and squish resize approaches? 
When might you choose one over the others?

> 16) What is data augmentation? Why is it needed?

Data augmentation is the process of increasing the diversity and amount of training data without collecting new data. This is done by applying various transformations (e.g. rotation, flipping, perspective warping, brightness changes and contrast changes) such that they appear different, but do not actually change the meaning of the existing dataset. 

This is useful as image labelling can be slow and expensive. Furthermore, it enables the model to better *generalize* the basic concept of what an object is and how the objects of interest are represented in images (in practice, by increasing the variance of  data, the model becomes less likely to memorize the features of the training).

> 17) What is the difference between item_tfms and batch_tfms?

> 18) What is a confusion matrix?

A representation of the predictions made versus the correct labels. The rows of the matrix represent the actual labels while the columns represent the predictions. Therefore, the diagonal of the matrix shows the images which were classified correctly, and the off-diagonal cells represent those which were classified incorrectly.

After inspecting the confusion matrix, it's helpful to see where exactly our errors are occurring, to see whether they're due to a dataset problem (e.g., images that might be labeled incorrectly), or a model problem (perhaps it isn't handling images taken with unusual lighting, or from a different angle). To do this, we can then sort our images by their *loss*.

> 19) What does export save?

> 20) What is it called when we use a model for getting predictions, instead of training?

Inference.

> 21) What are IPython widgets?

> 22) When might you want to use CPU for deployment? When might GPU be better?

> 23) What are the downsides of deploying your app to a server, instead of to a 
client (or edge) device such as a phone or PC?

> 24) What are three examples of problems that could occur when rolling out a bear 
warning system in practice?

> 25) What is "out-of-domain data"?

> 26) What is "domain shift"?

> 27) What are the three steps in the deployment process?