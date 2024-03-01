# Lesson4 Quiz

## Neural Net Foundations

> 1. How is a grayscale image represented on a computer? How about a color image?
> 1. How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?
> 1. Explain how the "pixel similarity" approach to classifying digits works.
> 1. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
> 1. What is a "rank-3 tensor"?
> 1. What is the difference between tensor rank and shape? How do you get the rank from the shape?

> 7. What are RMSE and L1 norm?

To have a sense of whether our model fits into data, we need a numeric measure to infer the difference between our prediction and the actual value. 

The most common measures are:
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

1. What is SGD?
1. Why does SGD use mini-batches?
1. What are the seven steps in SGD for machine learning?
1. How do we initialize the weights in a model?
1. What is "loss"?
1. Why can't we always use a high learning rate?
1. What is a "gradient"?
1. Do you need to know how to calculate gradients yourself?
1. Why can't we use accuracy as a loss function?
1. Draw the sigmoid function. What is special about its shape?
1. What is the difference between a loss function and a metric?
1. What is the function to calculate new weights using a learning rate?
1. What does the `DataLoader` class do?
1. Write pseudocode showing the basic steps taken in each epoch for SGD.
1. Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?
1. What does `view` do in PyTorch?
1. What are the "bias" parameters in a neural network? Why do we need them?
1. What does the `@` operator do in Python?
1. What does the `backward` method do?
1. Why do we have to zero the gradients?
1. What information do we have to pass to `Learner`?
1. Show Python or pseudocode for the basic steps of a training loop.
1. What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.
1. What is an "activation function"?
1. What's the difference between `F.relu` and `nn.ReLU`?
1. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?