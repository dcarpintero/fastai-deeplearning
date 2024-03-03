# Lesson4 Quiz

## Neural Net Foundations

> 1. How is a grayscale image represented on a computer? How about a color image?

Images are represented by arrays with pixel values representing the content of the image. For greyscale images, a 2-dimensional array is used with the pixels representing the greyscale values, with a range of 256 integers. A value of 0 would represent white, and a value of 255 represents black, and different shades of greyscale in between. For color images, three color channels (red, green, blue) are typicall used, with a separate 256-range 2D array used for each channel. A pixel value of 0 again represents white, with 255 representing solid red, green, or blue. The three 2-D arrays form a final 3-D array (rank 3 tensor) representing the color image.

> 2. How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?

There are two subfolders, train and valid, the former contains the data for model training, the latter contains the data for validating model performance after each training step. Evaluating the model on the validation set serves two purposes: a) to report a human-interpretable metric such as accuracy (in contrast to the often abstract loss functions used for training), b) to facilitate the detection of overfitting by evaluating the model on a dataset it hasn’t been trained on (in short, an overfit model performs increasingly well on the training set but decreasingly so on the validation set). 

> 3. Explain how the "pixel similarity" approach to classifying digits works.

In the “pixel similarity” approach, we generate an archetype for each class we want to identify. In our case, we want to distinguish images of 3’s from images of 7’s. We define the archetypical 3 as the pixel-wise mean value of all 3’s in the training set. Analoguously for the 7’s. You can visualize the two archetypes and see that they are in fact blurred versions of the numbers they represent.
In order to tell if a previously unseen image is a 3 or a 7, we calculate its distance to the two archetypes (here: mean pixel-wise absolute difference). We say the new image is a 3 if its distance to the archetypical 3 is lower than two the archetypical 7.

> 4. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.

It is a concise Pythonic way to create lists.

```python
[x*2 for _ in range(10) if x%2==1]
```

> 5. What is a "rank-3 tensor"?

A tensor with 3 dimensions. An easy way to identify the rank is the number of indices you would need to reference a number within a tensor. A scalar can be represented as a tensor of rank 0 (no index), a vector can be represented as a tensor of rank 1 (one index, e.g., v[i]), a matrix can be represented as a tensor of rank 2 (two indices, e.g., a[i,j]), and a tensor of rank 3 is a cuboid or a “stack of matrices” (three indices, e.g., b[i,j,k]). In particular, the rank of a tensor is independent of its shape or dimensionality, e.g., a tensor of shape 2x2x2 and a tensor of shape 3x5x7 both have rank 3.
Note that the term “rank” has different meanings in the context of tensors and matrices (where it refers to the number of linearly independent column vectors).

> 6. What is the difference between tensor rank and shape? How do you get the rank from the shape?

The tensor `rank` is the number of dimensions, whereas the shape is the size of each dimension.

> 7. What are RMSE and L1 norm?

To have a sense of whether our model fits into data, we need a numeric measure to infer the difference between our prediction and the actual value. The most common measures are:
- **Mean Absolute Difference (MAE)**, L1 norm, which adds the absolute values of the difference.
- **Root Mean Square Error (RMSE)**, L2 norm, which takes the mean of the square (makes everything positive) and then takes the square root (undoes squaring).

> 8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?

Using PyTorch tensors which are optimized to run on GPUs. Numpy is also an option since both run on a compiled object written (and optimized) in another language—specifically such as C++ (or Rust).

See also [JAX](https://github.com/google/jax).

> 9. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.

```python
t = torch.Tensor(range(1,10)).view(3, 3)
t = t*2
t[1:,1:]
```

> 10. What is broadcasting?

Broadcasting is a PyTorch and Numpy mechanism that enables to perform operations between arrays/tensors of different shapes. In practice, the smaller tensor is "broadcast" across the larger tensor so that they have compatible shapes for element-wise operations.

Note that this is purely an abstract concept, PyTorch doesn't actually allocate the broadcasted tensor in memory.

> 11. Are metrics generally calculated using the training set, or the validation set? Why?

Metrics are normally calculated using the validation set, which contains data that is isolated from the training process.

> 12. What is SGD? 

Stocastic Gradient Descent (SDG) is an optimization algorithm used in deep learning. In practice, it corresponds to the vision from Arthur Samuel:

> *Suppose we arrange for some automatic means of testing the effectiveness of any current weight assignment in terms of actual performance and provide a mechanism for altering the weight assignment so as to maximize the performance. We need not go into the details of such a procedure to see that it could be made entirely automatic and to see that a machine so programmed would "learn" from its experience.*

Specifically, SGD updates the parameters of a model in order to minimize a given loss function that was evaluated on the predictions and target. The main idea behind SGD (and many optimization algorithms, for that matter) is that the gradient of the loss function provides an indication of how that loss function changes in the parameter space, which we can use to determine how best to update the parameters in order to minimize the loss function.

> 13. Why does SGD use mini-batches?

The use of mini-batches has several advantages: (i) **computational efficiency** (GPUs only performs well if they have lots of work to do at a time), (ii) **better generalization** (rather than simply enumerating our dataset in order for every epoch, we randomly shuffle it on every epoch to introduce variance), and (iii) **memory usage** (it is a practical choice to not overload GPU's memory). 

> 14. What are the seven steps in SGD for machine learning?

- initialize parameters.
- calculate predictions on a mini-batch.
- calculate the average loss between the predictions and the targets in the mini-batch.
- calculate the gradients, this provides an indication of how the parameters need to change to minimize the loss function.
- step the weights based on the gradients and learning rate.
- repeat from step (ii).
- stop once a condition is met (e.g. time constraint or based on when the training/validation losses and metrics stop improving).

> 15. How do we initialize the weights in a model?

They are typically initiliazed randomly.

> 16. What is "loss"?

A loss measures the difference between the predictions of the model and the targets:

- Mean Absolute Difference (L1 norm).
- Root Mean Square Error (RMSE or L2 norm). In practice, this choice will penalize bigger mistakes more heavily.

> 17. Why can't we always use a high learning rate?

A high learning rate might prevent the model from converging.

> 18. What is a "gradient"?

A measure inferred from the derivative of a function that indicates how the a function output would change with changes of the parameters of the model. In our case, it tells us how much we have to change each weight to make our model better.

> 19. Do you need to know how to calculate gradients yourself?

This is normally handed over to PyTorch. If `requires_grad=True`, the gradients can be returned by calling the `backward` method.

> 20. Why can't we use accuracy as a loss function?

> 21. Draw the sigmoid function. What is special about its shape?

> 22. What is the difference between a loss function and a metric?

A loss function represents the difference between the predictions and the target, and it is used by the computer model to automatically update its weights; whereas a metric provides an indication to the computer programmer of how well the model is performing, driving therefore human understanding.

> 23. What is the function to calculate new weights using a learning rate?

The optimizer step function, e.g.:

```python
weights -= grad * lr
```

> 24. What does the `DataLoader` class do?

The DataLoader class can take any Python collection and turn it into an iterator over many batches.

> 25. Write pseudocode showing the basic steps taken in each epoch for SGD.

```python
for x,y in dl:
    preds = model(x)
    loss = criterion(preds, y)
    loss.backward()
    params -= params.grad * lr
```

> 26. Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?

```python
def f(x, y):
    return list(zip(x, y))
```

This data structure is useful for machine learning models when you need lists of tuples where each tuple would contain input data and a label.

> 27. What does `view` do in PyTorch?

Modifies the shape of a tensor without changing its content.

> 28. What are the "bias" parameters in a neural network? Why do we need them?

The "bias" parameters in a neural network are additional parameters added to each neuron in a layer that are used to shift the output of the neuron's activation function along the y-axis. They work in conjunction with the weights to determine the strength of the neuron's output.

Without bias, a neuron's output would always be zero when the input is zero, regardless of the weights. This limits the types of functions that the network can represent. By adding a bias term, we allow the neuron to output non-zero values when the input is zero, which increases the range of functions that the network can model. In other words, bias parameters allow neural networks to represent patterns that do not necessarily pass through the origin of the coordinate system, **making them more flexible and capable of learning complex patterns**.

> 29. What does the `@` operator do in Python?

Matrix multiplication.

> 30. What does the `backward` method do?

This method returns the current gradients.

> 31. Why do we have to zero the gradients?

PyTorch will add the gradients of a variable to any previously stored gradients. If the training loop function is called multiple times, without zeroing the gradients, the gradient of current loss would be added to the previously stored gradient value.

> 32. What information do we have to pass to `Learner`?

We need to pass in the DataLoaders, the model, the optimization function, the loss function, and optionally any metrics to print.

> 33. Show Python or pseudocode for the basic steps of a training loop.

```python
def train_epoch(model, lr, params):
    for xa, xb in df:
        preds = model(xa)
        loss = criterion(preds, xb)
        loss.backward()
        
        for p in params:
            p.data -= p.grad * lr
            p.grad.zero_()

for i in range(20):
    train_epoc(model, lr, params)
```

> 34. What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.

ReLU stands for rectified linear unit, it is an activation function that converts negative values to 0, while maintaining possitive values.

> 35. What is an "activation function"?

The activation function is another function that is part of the neural network, which has the purpose of providing non-linearity to the model. The idea is that without an activation function, we just have multiple linear functions of the form y=mx+b. However, a series of linear layers is equivalent to a single linear layer, so our model can only fit a line to the data. By introducing a non-linearity in between the linear layers, this is no longer true. Each layer is somewhat decoupled from the rest of the layers, and the model can now fit much more complex functions. In fact, it can be mathematically proven that such a model can solve any computable problem to an arbitrarily high accuracy, if the model is large enough with the correct weights. This is known as the universal approximation theorem.

> 36. What's the difference between `F.relu` and `nn.ReLU`?

`F.relu` refers to the Python implementation of the relu activation function, whereas `nn.ReLU` is a PyTorch module.

> 37. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?

Using multiple nonlinear functions enables to approximate complex patterns such as those needed for computer vision or speech recognition. There are also practical performance benefits to using more than one nonlinearity. We can use a deeper model with less number of parameters, better performance, faster training, and less compute/memory requirements.