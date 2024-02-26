# Lesson 4

[Notebook](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work)

## How does a  neural net really work?

- In a model, we typically look for: how fast are they, how much memory do they use, and how accurate are they? 
- The goal of a ML model is to fit functions to data.
- To have a sense of whether our fit is appropiate, we need a numeric measure. An easy metric is **mean absolute error**, a loss function which is the distance from each data point to the curve.
- In order to automate this process, we can use `derivates`, a concept in calculus that measures how a function output changes as its input changes (this indicates whether the parameters of an AI model should increase or decrease). A generalization of the derivative for functions with multiple inputs is the `gradient`, which represents a vector pointing in the direction of the greatest rate of increase.
- Once we know the `gradients` for our (randomly initiliazed) parameters, we can adjust them by a `learning rate`. In practice, our learning rate descreases as we train. This is done using a `learning rate schedule`, and can be automated in most deep learning frameworks, such as fastai and PyTorch.

## How a neural network approximates any given function

A neural network is an infinitely expressive function, that can approximate any computable function **given enough time and enough data** (e.g. NLP, computer vision, speech recognition), given enough parameters. See [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

The way a neural network approximates a function actually turns out to be very simple. The key trick is to combine two extremely basic steps:

1. Matrix multiplication, which is just multiplying things together and then adding them up
2. The function 𝑚𝑎𝑥(𝑥,0), which simply replaces all negative numbers with zero.

In PyTorch, the function  𝑚𝑎𝑥(𝑥,0) is written as np.clip(x,0). The combination of a linear function and this max() is called a rectified linear function, and it can be implemented like this:

```python
def rectified_linear(m,b,x):
    y = m*x+b
    return torch.clip(y, 0.) 
    # return F.relu(m*x+b)
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

### Arrays and Tensors

### Broadcasting

### Stochastic Gradient Descent (SGD)

### Loss Function

### Mini Batches

## Books

- [Python for Data Analysis](https://wesmckinney.com/book/)

## References

- [How does a  neural net really work?](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work)
- [Which image models are best?](https://www.kaggle.com/code/jhoward/which-image-models-are-best/)
- [Calculus](https://www.youtube.com/playlist?list=PLybg94GvOJ9ELZEe9s2NXTKr41Yedbw7M)
