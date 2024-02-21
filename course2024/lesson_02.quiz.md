# Lesson2 Quiz

## From Model to Production, and the State of Deep Learning

> 1) Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.

- clean images in training vs. blurry, motion, distant, covered bear images in prod.
- daylight images in training vs. night images in prod (and viceversa).
- the training set is biased towards a certain feature (e.g. color, type).
- unreal bear images (e.g. from comic characters) in the training dataset.

*To mitigate these issues, it is useful to recognize when unexpected image types arise in production (checking for *out-of-domain* data).*

> 2) Where do text models currently have a major deficiency?

- lack of ground truth and up-to-date data (*partially mitigated with RAG, fine-tuning, temperature adjustment, and large context windows*).
- struggle to work in multilingual mode (*Cohere just released Aya on Feb 2024 https://txt.cohere.com/aya/*).

> 3) What are possible negative societal implications of text generation models?

- LLMs might generate highly compelling responses that might deceive readers.

*Some initiatives to migitate this risk include alignment with
principles that humans find agreeable (see e.g. [Constitutional AI](https://cdn.sanity.io/files/4zrzovbb/website/7512771452629584566b6303311496c262da1006.pdf) initiative by Anthropic)*.

> 4) In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?

Where possible, the first step is to use an entirely manual process, with **the deep learning model running in parallel but not being used directly to drive any actions**. The humans involved in the manual process should look at the deep learning outputs and check whether they make sense.

The second step would be to try to limit the scope of the model, and have it carefully supervised by people. For instance, by doing a **small geographically and time-constrained trial of the model-driven approach**.

> 5) What kind of tabular data is deep learning particularly good at?

 Columns containing **natural language**, and **high-cardinality** (i.e., something that contains a large number of discrete choices, such as zip code or product ID). 

*On the down side, deep learning would generally take longer to train than random forests or gradient boosting machines, although this is changing thanks to libraries such as [RAPIDS](https://rapids.ai/), which provides GPU acceleration for the whole modeling pipeline.*

> 6) What's a key downside of directly using a deep learning model for recommendation systems?

- **Bias amplification**: For example, if certain types of content are historically more popular among a specific user group, the model might prioritize recommending similar content to similar users, further entrenching those preferences and biases.

- **Reduced content diversity**: By focusing on optimizing user engagement or click-through rates, deep learning-based recommendation systems can create a [filter bubble](http://www.ted.com/talks/eli_pariser_beware_online_filter_bubbles.html), where users are continually presented with content that aligns with their existing preferences. This can reduce exposure to diverse viewpoints and information, limiting users' awareness of different perspectives.

- **Cold Start Problem**: New users or items with little to no interaction data present challenges for deep learning models, which rely heavily on historical data to make recommendations.

> 7) What are the steps of the Drivetrain Approach?

The Drivetrain Approach is a framework designed to **use data not just to generate more data (in the form of predictions), but to produce actionable outcomes**:

- **Define the Objective**: Start with the end goal of your project (e.g. what is the user’s main objective in typing in a search query?).
- **Determine the Levers**: Identify actions that can be taken to influence the final outcome (e.g. control the ranking of the search results).
- **Understand the new Data** needed to implement the levers (e.g. the implicit information regarding which pages linked to which other pages could be used for this purpose).
- **Build the Model**: Develop a model to predict the best actions for achieving your goals (e.g. Larry Page and Sergey Brin invented the graph traversal algorithm PageRank and built an engine on top of it that revolutionized search).

*Optimizing for an actionable outcome over the right predictive models can be a company’s most important strategic decision. See also [Designing Great Data Products](https://www.oreilly.com/radar/drivetrain-approach-data-products/)*

> 8) How do the steps of the Drivetrain Approach map to a recommendation system?

- **Objective**: drive additional sales by surprising and delighting the customer with books he or she would not have purchased without the recommendation.
- **Levers**: the ranking of recommendations.
- **Data**:  conducting many randomized experiments in order to collect data about a wide range of recommendations for a wide range of customers.
- **Model**: build two models that predict purchase probabilities based on whether a recommendation is seen or not. The utility of a recommendation is assessed by the difference in these probabilities, aiding in avoiding redundant or ineffective suggestions. A Simulator tests the utility of various books, and an Optimizer then ranks these recommendations.

> 9) Create an image recognition model using data you curate, and deploy it on the web.

https://huggingface.co/spaces/dcarpintero/interstellar

> 10) What are DataLoaders?

`DataLoaders` is a class that stores any DataLoader objects passed to it, making them accessible as train and valid datasets.

> 11) What four things do we need to tell fastai to create DataLoaders?

- What kinds of data you are working with: specify the type of data, such as images or texts.
- How to get the list of items: define a method for retrieving the items (e.g., file paths to images) that will be used for training and validation.
- How to label these items: specify a way to label each item based on your dataset.
- How to create the validation set: define a strategy for splitting the data into training and validation sets to evaluate the model's performance on unseen data.

> 12) What does the splitter parameter to DataBlock do?

The `splitter` parameter in DataBlock is used to define how the dataset is split into training and validation sets. In the provided example, the `RandomSplitter` function is used with the splitter parameter, indicating that the dataset should be randomly split. The `valid_pct` argument specifies the percentage of the dataset to be used as the validation set (20% in this case), and the `seed` argument ensures that the same split is reproducible across different runs by fixing the seed for the random number generator. 

> 13) How do we ensure a random split always gives the same validation set?

 By specifying a `seed` argument in the RandomSplitter function (or any other splitter function that involves randomness), you guarantee that the random split is reproducible. 

> 14) What letters are often used to signify the independent and dependent variables?

x (independent), and y (dependent).

> 15) What's the difference between the crop, pad, and squish resize approaches? 
When might you choose one over the others?

- `crop` is the `default Resize()` method, and it crops the images to fit a square shape of the size requested, using the full width or height. This can result in losing some important details. 
- `pad` is an `alternative Resize()`, which pads the matrix of the image’s pixels with zeros (which shows as black when viewing the images). If we pad the images then we have a whole lot of empty space, which is just wasted computation for our model, and results in a lower effective resolution for the part of the image we actually use.
- `squish` is another `alternative Resize()`, which can either squish or stretch the image. This can cause the image to take on an unrealistic shape, leading to a model that learns that things look different to how they actually are, which we would expect to result in lower accuracy.

Which resizing method to use therefore depends on the underlying problem and dataset. For example, if the features in the dataset images take up the whole image and cropping may result in loss of information, squishing or padding may be more useful.

> 16) What is data augmentation? Why is it needed?

`Data augmentation` is the process of increasing the diversity and amount of training data without collecting new data. This is done by applying various transformations (e.g. rotation, flipping, perspective warping, brightness changes and contrast changes) such that they appear different, but do not actually change the meaning of the existing dataset. 

This is useful as image labelling can be slow and expensive. Furthermore, it enables the model to better *generalize* the basic concept of what an object is, and how the objects of interest are represented in images (in practice, by increasing the variance of  data, the model becomes less likely to memorize the features of the training).

> 17) What is the difference between item_tfms and batch_tfms?

- `item_tfms` are transformations applied to a single data sample on the CPU. Resize() is a common transform because the mini-batch of input images to a cnn must have the same dimensions.

- `batch_tfms` are applied to batched data samples (aka individual samples that have been collated into a mini-batch) on the GPU. They are faster and more efficient than item_tfms. A good example of these are the ones provided by aug_transforms(). Inside are several batch-level augmentations that help many models.

> 18) What is a confusion matrix?

A representation of the predictions made versus the correct labels. The rows of the matrix represent the actual labels while the columns represent the predictions. Therefore, the diagonal of the matrix shows the images which were classified correctly, and the off-diagonal cells represent those which were classified incorrectly.

After inspecting the confusion matrix, it's helpful to see where exactly our errors are occurring, to see whether they're due to a dataset problem (e.g., images that might be labeled incorrectly), or a model problem (perhaps it isn't handling images taken with unusual lighting, or from a different angle). To do this, we can then sort our images by their *loss*.

> 19) What does export save?

`export` saves both the `architecture`, as well as the `trained parameters` of the neural network architecture. It also saves how the DataLoaders are defined.

> 20) What is it called when we use a model for getting predictions, instead of training?

Inference.

> 21) What are IPython widgets?

IPython widgets are JavaScript and Python combined functionalities that let us build and interact with GUI components directly in a Jupyter notebook. An example of this would be an upload button, which can be created with the Python function widgets.FileUpload().

> 22) When might you want to use CPU for deployment? When might GPU be better?

GPUs are best for doing identical work in parallel. If you will be analyzing single pieces of data at a time (like a single image or single sentence), then CPUs may be more cost effective. 

GPUs could be used if you collect user responses into a batch at a time, and perform inference on the batch. This may require the user to wait for model predictions. Additionally, there are many other complexities when it comes to GPU inference, like memory management and queuing of the batches.

> 23) What are the downsides of deploying your app to a server, instead of to a 
client (or edge) device such as a phone or PC?

- latency
- operational costs
- data privacy

> 24) What are three examples of problems that could occur when rolling out a bear warning system in practice?

- Handling night-time images
- Dealing with low-resolution images (ex: some smartphone images)
- The model returns prediction too slowly to be useful

> 25) What is "out-of-domain data"?

Data that falls outside the range or scope of the data on which a model was trained. 

> 26) What is "domain shift"?

This is when the type of data changes gradually over time. For example, an insurance company is using a deep learning model as part of their pricing algorithm, but over time their customers will be different.

> 27) What are the three steps in the deployment process?

- **Manual Process First**: using the model's outputs for verification without driving actions directly. Example: Park rangers monitoring video feeds for bear sightings highlighted by the model, enhancing alertness without relying solely on the model.
  
- **Limited Scope Deployment**: deploy the model in a controlled, small-scale environment to closely monitor its effectiveness and manage risks. Example: Testing the bear classifier at a single observation post over a week, with park rangers verifying each alert.

- **Gradual Expansion**: Slowly expand the deployment area, ensuring robust reporting systems are in place to detect any significant deviations or issues. Example: Observing changes in the frequency of bear alerts and incorporating measures in regular reports to identify potential problems during rollout expansion.