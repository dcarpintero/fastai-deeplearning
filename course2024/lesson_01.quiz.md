
> 1. What was the name of the first device that was based on the principle of the artificial neuron?

Mark I Perceptron, a device capable of learning and recognizing simple shapes, built by Frank Rosenblatt. 

> 2. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?

- A set of processing units
- A state of activation
- An output function for each unit
- A pattern of connectivity among units
- A propagation rule for propagating patterns of activities through the network of connectivities
- An activation rule for combining the inputs impinging on a unit with the current state of that unit to produce a new level of activation for the unit
- A learning rule whereby patterns of connectivity are modified by experience
- An environment within which the system must operate

> 3. What were the two theoretical misunderstandings that held back the field of neural networks?

In 1969, Marvin Minsky and Seymour Papert demonstrated in their book, “Perceptrons”, that a single layer of artificial neurons cannot learn simple, critical mathematical functions like XOR logic gate. While they subsequently demonstrated in the same book that additional layers can solve this problem, only the first insight was recognized, leading to the start of the first AI winter.

In the 1980’s, models with two layers were being explored. Theoretically, it is possible to approximate any mathematical function using two layers of artificial neurons. However, in practice, these networks were too big and too slow. While it was demonstrated that adding additional layers improved performance, this insight was not acknowledged, and the second AI winter began. In this past decade, with increased data availability, and improvements in computer hardware (both in CPU performance but more importantly in GPU performance), neural networks are finally living up to its potential.

> 4. What is a GPU?

GPU stands for Graphics Processing Unit .

> 5. Why is it hard to use a traditional computer program to recognize objects in an image?

Tradicional computer programs require to specify rules in form of computer instructions to carry out a certain task. In the present case, ideating a set of rules to recognize objets in an image is a highly complex task.

> 6. What did Samuel mean by "weight assignment"? What term do we normally use in deep learning for what Samuel called "weights"?

The modern terminology refers to what Samuel called "weights" as model parameters, with the term "weights" now denoting a specific type of parameter.

> 7. Why is it hard to understand why a deep learning model makes a particular prediction?

This is a highly-researched topic known as interpretability of deep learning models. Deep learning models are hard to understand in part due to their “deep” nature.

> 8. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?

The universal approximation theorem.

> 9. What do you need in order to train a model?

- Architecture of the model such as number of layers, and activation units.
- A function that randomly assigns weights to the model.
- Labelled data.
- A function that infers the performance of the model (loss function).
- A function that updates the weights of the model according to the performace (optimizer).

> 10. How could a feedback loop impact the rollout of a predictive policing model?

A feedback loop might amplify bias.

> 11. Do we always have to use 224×224-pixel images with the cat recognition model?

224x224 is commonly used for historical reasons. You can increase the size and get better performance, but at the price of speed and memory consumption.

> 12. What is the difference between classification and regression?

In regression the output is a continuous value (e.g. the price of a house), whereas in classification the output is among a discret set of labels (e.g. the types of houses).

> 13. What is a validation set? What is a test set? Why do we need them?

A validation set is used to evaluate the performance of a model during training to prevent overfitting. This ensures that the model performance is not due to memorization of the dataset, but rather because it learns the appropriate features to use for prediction. However, it is possible that we overfit the validation data as well due to the fact that the human modeler is also part of the training process, adjusting hyperparameters and said training procedures according to the validation performance. 

Therefore, another unseen portion of the dataset, the test set, is used for final evaluation of the model. This splitting of the dataset is necessary to ensure that the model generalizes to unseen data.

> 14. What will fastai do if you don't provide a validation set?

fastai will automatically create a validation dataset. It will randomly take 20% of the data and assign it as the validation set ( valid_pct = 0.2 ).

> 15. Does it make sense to always use a random sample for a validation set? Why or why not?

A good validation or test set should be representative of new data you will see in the future. In certain cases such as time series data it is not appropiate to use random samples, but rather defining different time periods for the train, valudation, and test sets.

> 16. What is overfitting? Provide an example.

Overfitting refers to *memorize* the features of the training set such that the model yields a good performance, but generalizes poorly in prodution. This situation might happen when training a model for a high number of epochs.

> 17. What is a metric? How does it differ from "loss"?

A *metric* is a value directed to the human modeler to indicate the performance of a model after training, whereas the *loss* is a value intended to the computerized training process that determines the difference between the prediction and the actual result for the purpose of updating the weights of the model.

> 18. How can pretrained models help?

Pretrained models can be leveraged for downstream tasks, reducing training efforts. For example, pretrained models in computer vision are already prepared to recognize commonplace features such as edges, lines and basic shapes.

> 19. What is the "head" of a model?

This is usually the last layer (or set of layers) of the model used for the intended original task. In practice, during transfer learning this layer is replaced by random weights and trained from scratch to adapt the model to a downstream task.

> 20. What kinds of features do the early layers of a CNN find? How about the later layers?

Early layers learn basic and generic features that are common to images such as: edges and corners, and textures. The filters (kernels) in these layers activate in response to these simple features wherever they occur in the input image. The activation maps generated at this stage highlight the presence and location of these basic features.

Later layers combine the basic features detected by earlier layers into more complex and abstract representations. These features are more specific to the dataset and the task for which the CNN has been trained such as object parts and entire objects.

This hierarchical feature detection mechanism is what enables CNNs to learn from visual data and generalize well across similar tasks.

> 21. Are image models only useful for photos?

Various types of information can be represented as images. For example, a sound can be converted into a spectrogram, which is a visual interpretation of the audio. Time series (e.g. financial data) can also be converted to image by plotting on a graph.

> 22. What is an "architecture"?

The architecture is the template or structure of the model we are trying to fit. It defines the mathematical model we are trying to fit.

> 23. What is segmentation?

A pixelwise classification problem, where we attempt to predict a label for every single pixel in the image. This provides a mask for which parts of the image correspond to the given label.

> 24. What is `y_range` used for? When do we need it?

`y_range` is being used to limit the values predicted when our problem is focused on predicting a numeric value in a given range (ex: predicting movie ratings, range of 0.5-5).

> 25. What are "hyperparameters"?

Hyperparameters are the configuration settings used to structure and tune the training of a model:
- number of layers and units
- number of epochs
- batch size
- learning rate

The process of selecting the optimal hyperparameters is known as hyperparameter tuning or optimization (see e.g. https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html). 

> 26. What’s the best way to avoid failures when using AI in an organization?

- Make sure a training, validation, and testing set is defined to evaluate the model in an appropriate manner.
- Try out a simple baseline, which future models should hopefully beat. Or even this simple baseline may be enough in some cases.