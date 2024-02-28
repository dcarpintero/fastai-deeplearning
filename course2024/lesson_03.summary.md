# Lesson 4

## Historical Notes
- In 1974, Paul Werbos invented back-propagation for NN ([Werbos 1994](https://books.google.com/books/about/The_Roots_of_Backpropagation.html?id=WdR3OOM2gBwC)). His development was almost entirely ignored for decades, but today it is considered the most important foundation of modern AI.
- In 2018, Yann Lecun, Yoshua Bengio, and Geoffrey Hinton, were awarded the highest honor in computer science, the Turing Award (generally considered the "Nobel Prize of computer science").
- Jurgen Schmidhuber (who many believe should have shared in the Turing Award) pioneered many important ideas, including working with his student Sepp Hochreiter on the long short-term memory (LSTM) architecture (widely used for speech recognition and other text modeling tasks).

## How does a neural net really work?

- In a model, we typically look for: how fast are they, how much memory do they use, and how accurate are they? 
- The goal of a ML model is to fit mathematical functions to data.
- To have a sense of whether our fit is appropiate, we need a numeric measure. An easy metric is **mean absolute error**, a loss function which is the distance from each data point to the function output.
- In order to automate this process, we can use `derivates`, a concept in calculus that measures how a function output changes as its input changes (this indicates whether the parameters of an AI model should increase or decrease). A generalization of the derivative for functions with multiple inputs is the `gradient`, which represents a vector pointing in the direction of the greatest rate of increase.
- Once we know the `gradients` for our (randomly initiliazed) parameters, we can adjust them by a `learning rate`. In practice, our learning rate descreases as we train. This is done using a `learning rate schedule`, and can be automated in most deep learning frameworks, such as fastai and PyTorch.

## How a neural network approximates any given function

A neural network is an infinitely expressive function, that can approximate any computable function **given enough time and enough data** (e.g. NLP, computer vision, speech recognition), **and enough parameters**. See [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

The way a neural network approximates a function actually turns out to be very simple. The key trick is to combine two extremely basic steps:

1. **Matrix multiplication**, which is just multiplying things together and then adding them up
2. The function **𝑚𝑎𝑥(𝑥, 0)**, which simply replaces all negative numbers with zero.

In PyTorch, the function  𝑚𝑎𝑥(𝑥,0) is written as np.clip(x,0). The combination of a linear function and this max() is called a rectified linear function, and it can be implemented like this:

```python
# F.relu(m*x+b)
def rectified_linear(m,b,x):
    y = m*x+b
    return torch.clip(y, 0.)   
```

With enough of rectified linear functions added together, we could approximate any function with a single input, to whatever intended accuracy.

## Python and PyTorch Tips

- To fix values passed to a function in python, we use the `partial` function:

```python
def quad(a, b, c, x):
    return a*x**2 + b*x + c

def mk_quad(a,b,c): 
    return partial(quad, a,b,c)

f = mk_quad(3,2,1)
# 3𝑥2 + 2𝑥 + 1
```

- To interact with jupyter notebooks use `@interact`:

```python
@interact(a=1.1, b=1.1, c=1.1)
def plot_quad(a, b, c):
    plt.scatter(x,y)
    plot_function(mk_quad(a,b,c), ylim=(-3,13))
```

- Use the `*` operator to unpack the elements of a `variable`. For example, if `params` is `[a, b, c]`, then `f(*params)` is equivalent to `f(a, b, c)`.

```python
def f_packed_Args(params): 
    f(*params) # f(a, b, c)
```

- To tell PyTorch that we want to calculate gradients, we need to use the `requires_grad_()` attribute:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
x.requires_grad_()

# Now any operation involving x will be tracked for gradient computation
y = x * 2
z = y.mean()

# Calculate gradients
z.backward()

# Gradients are stored in x.grad
print(x.grad)  # prints: tensor([0.6667, 0.6667, 0.6667])
```

## Matrix Multiplication

In practice, both the inputs and the weights of a NN can be represented as a matrices, such that the computation of the outputs for all inputs can be performed in a single matrix multiplication.

## Numpy Arrays and Tensors

Although NumPy provides similar functionality to Tensors, it does not support GPU operations and gradients (which are critical for DL).

*Python is slow compared to many languages. Anything fast in Python, NumPy, or PyTorch is likely to be a wrapper for a compiled object written (and optimized) in another language—specifically C. In fact, **NumPy arrays and PyTorch tensors can finish computations many thousands of times faster than using pure Python.***

```python
# create array/tensor
data = [[1,2,3],[4,5,6]]
arr = array (data)
tns = tensor(data)
```

## Broadcasting

Broadcasting allows to perform operations on tensors of different shapes. The smaller tensor is "broadcast" across the larger tensor so that they have compatible shapes for element-wise operations.

```python
tensor([1,2,3]) + tensor(1)
```

Note that broadcasting is a conceptual tool; PyTorch doesn't actually allocate the broadcasted tensor in memory. It does the whole calculation in C (or, if you're using a GPU, in CUDA, the equivalent of C on the GPU), tens of thousands of times faster than pure Python (up to millions of times faster on a GPU).

## Stochastic Gradient Descent (SGD)

As described by Arthur Samuel:

> *Suppose we arrange for some automatic means of testing the effectiveness of any current weight assignment in terms of actual performance and provide a mechanism for altering the weight assignment so as to maximize the performance. We need not go into the details of such a procedure to see that it could be made entirely automatic and to see that a machine so programmed would "learn" from its experience.*

For an exemplary image classifier, here are the steps that we are going to require:

1. *Initialize* the weights.
1. For each image, use these weights to *predict* whether it appears to be a 3 or a 7.
1. Based on these predictions, calculate how good the model is (its *loss*).
1. Calculate the *gradient*, which measures for each weight, how changing that weight would change the loss.
1. *Step* (that is, change) all the weights based on that calculation.
1. Go back to the step 2, and *repeat* the process.
1. Iterate until you decide to *stop* the training process (for instance, because the model is good enough or you don't want to wait any longer).

There are many different ways to do each of these seven steps, and we will be learning about them throughout the rest of this book.  Here are a few guidelines:

- **Initialize**: We initialize the parameters to random values. This may sound surprising. There are certainly other choices we could make, such as initializing them to the percentage of times that pixel is activated for that category—but since we already know that we have a routine to improve these weights, it turns out that just starting with random weights works perfectly well.

- **Loss**: This is what Samuel referred to when he spoke of *testing the effectiveness of any current weight assignment in terms of actual performance*. We need some function that will return a number that is small if the performance of the model is good (the standard approach is to treat a small loss as good, and a large loss as bad, although this is just a convention).

- **Step**: A simple way to figure out whether a weight should be increased a bit, or decreased a bit, would be just to try it: increase the weight by a small amount, and see if the loss goes up or down. Once you find the correct direction, you could then change that amount by a bit more, and a bit less, until you find an amount that works well. However, this is slow! As we will see, the magic of calculus allows us to directly figure out in which direction, and by roughly how much, to change each weight, without having to try all these small changes. The way to do this is by calculating *gradients*. This is just a performance optimization, we would get exactly the same results by using the slower manual process as well.

- **Stop**: Once we've decided how many epochs to train the model for, we apply that decision. This is where that decision is applied. For our digit classifier, we would keep training until the accuracy of the model started getting worse, or we ran out of time.

### Calculating Gradients

The one magic step is the bit where we calculate the *gradients* to know how much we have to change each weight to make our model better. Remember that the *derivative* function indicates how much a change in its parameters will change its result. 

For instance, the derivative of the quadratic function at the value 3 tells us how rapidly the function changes at the value 3. In deep learning, *gradients* usually means the _value_ of a function's derivative at a particular argument value. The PyTorch API also puts the focus on the argument, not the function you're actually computing the gradients of:

```python
def f(x): 
    return x**2

xt = tensor(3.).requires_grad_()
yt = f(xt)
# calculate the gradients
yt.backward()
# prints tensor(6.) since the derivative of x**2 is 2*x
xt.grad
```

### Stepping with a Learning Rate

Deciding how to change our parameters based on the values of the gradients is an important part of the deep learning process. Nearly all approaches start with the basic idea of multiplying the gradient by some small number, called the *learning rate* (LR). The learning rate is often a number between 0.001 and 0.1, although it could be anything. Once you've picked a learning rate (using e.g. the *learning rate finder*), you can adjust your parameters using this simple function:

```python 
w -= gradient(w) * lr
```

### SGD and Mini-Batches

A mini-batch is a subset of the entire training dataset. Instead of passing the whole dataset through the network or a single example at a time, you pass a mini-batch of 'n' samples where 'n' is more than 1 but less than the total number of samples in the training set.

The use of mini-batches has several advantages: (i) **computational efficiency** (only perform well if they have lots of work to do at a time), (ii) **better generalization** (rather than simply enumerating our dataset in order for every epoch, we randomly shuffle it on every epoch to introduce variance), and (iii) **memory usage** (it is a practical choice to not overload GPU's memory). 

The size of the mini-batches is a hyperparameter of the model and can be tuned for best performance. A larger batch size means that you will get a more accurate and stable estimate of your dataset's gradients from the loss function, but it will take longer, and you will process fewer mini-batches per epoch. Common choices for the mini-batch size include 32, 64, and 128, but the optimal size can depend on the specific problem and the hardware used for training.

## Loss Function

- Mean Absolute Difference (L1 norm).
- Root Mean Square Error (RMSE or L2 norm). In practice, this choice will penalize bigger mistakes more heavily.

```python
import torch.nn.function as F

F.l1_loss(a, b)
# (a - b).abs().mean()
F.mse_loss(a, b)
# ((a - b)**2).mean().sqrt()

```

## Jargon Recap

| Term           | Meaning                                                                                                   |
| -------------- | --------------------------------------------------------------------------------------------------------- |
| ReLU           | Function that returns 0 for negative numbers and doesn't change positive numbers.                         |
| Mini-batch     | A small group of inputs and labels gathered together in two arrays. A gradient descent step is updated on this batch (rather than a whole epoch). |
| Forward pass   | Applying the model to some input and computing the predictions.                                           |
| Loss           | A value that represents how well (or badly) our model is doing.                                           |
| Gradient       | The derivative of the loss with respect to some parameter of the model.                                   |
| Backward pass  | Computing the gradients of the loss with respect to all model parameters.                                 |
| Gradient descent | Taking a step in the directions opposite to the gradients to make the model parameters a little bit better. |
| Learning rate  | The size of the step we take when applying SGD to update the parameters of the model.                     |
| Activations    | The outputs of the neurons in a layer of a neural network. These are computed by applying a function (usually non-linear) to the weighted sum of the inputs. |
| Parameters     | The weights and biases in a neural network that are learned during training. These define the transformation performed by the network on its inputs. |

## Books

- [Python for Data Analysis](https://wesmckinney.com/book/)

## References

- [How does a  neural net really work?](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work)
- [The best vision models for fine tuning](https://www.kaggle.com/code/jhoward/the-best-vision-models-for-fine-tuning)
- [Which image models are best?](https://www.kaggle.com/code/jhoward/which-image-models-are-best/)
- [Calculus](https://www.youtube.com/playlist?list=PLybg94GvOJ9ELZEe9s2NXTKr41Yedbw7M)
- [Matrix Multiplication](https://matrixmultiplication.xyz)
