## Deep Learning

Deep learning is a computer technique to extract and transform data–-with use cases ranging from human speech recognition to animal imagery classification–-by using multiple layers of neural networks. Each of these layers takes its inputs from previous layers and progressively refines them. The layers are trained by algorithms that minimize their errors and improve their accuracy. In this way, the network learns to perform a specified task.

## Neural Networks: A Brief History

- In 1943, neurophysiologist `Warren McCulloch` and logician `Walter Pitts` developed a mathematical model of an artificial neuron. Their paper, "A Logical Calculus of the Ideas Immanent in Nervous Activity", proposed using propositional logic to describe neural events due to the "all-or-none" nature of nervous activity. McCulloch and Pitts' model represented neurons with simple addition and thresholding.
- Psychologist `Frank Rosenblatt` expanded on their work by developing the `perceptron, a device capable of learning and recognizing simple shapes`.
- Marvin Minsky and Seymour Papert critiqued the perceptron in their book "Perceptrons", pointing out its `inability to learn certain functions like XOR`, and suggested multi-layer networks could overcome this limitation.
- The global academic community largely abandoned NN research for two decades following Minsky and Papert's critique.
- In 1986, the book "Parallel Distributed Processing (PDP)" by David Rumelhart, James McClellan, and `the PDP Research Group revitalized interest in neural networks by proposing a model closely resembling brain computation`.
- PDP defined key components of neural networks, including `processing units`, `activation states`, and `learning rules`, aligning with modern neural network architecture.
- Despite early challenges, NN evolved in the 1980s and 1990s with multi-layer models, overcoming previous limitations.
- Recent advancements in computing power, data availability, and algorithmic improvements have finally realized the potential of neural networks, fulfilling Rosenblatt's vision of machines capable of perception and recognition without human intervention.

## Machine Learning

- In 1949, IBM researcher `Arthur Samuel coined the term machine learning`, focusing on `teaching computers to learn from examples rather than programming explicit` instructions. Samuel critiqued traditional programming for its tediousness and proposed machine learning as a solution, where computers learn to solve problems by themselves.
- By 1961, his checkers-playing program demonstrated the effectiveness of machine learning by beating the Connecticut state champion.
- Samuel introduced the concept of `weight assignments` in machine learning, where weights are variables that influence the program's operation and outcomes.
He emphasized the need for an `automatic method to test and optimize these weight assignments` based on their `performance in real tasks`.
- The modern terminology refers to what Samuel called "weights" as model parameters, with the term "weights" now denoting a specific type of parameter.
- Samuel envisioned machine learning as a process where the adjustment of weights is automated based on performance, making the learning process entirely automatic.

## Neural Networks

- Neural networks serve as a highly flexible model, capable of solving virtually any problem to any desired accuracy level, as demonstrated by the universal approximation theorem.
- The process of training neural networks, or finding optimal weight assignments, is facilitated by a method known as `stochastic gradient descent (SGD)`.
SGD offers a `general approach to automatically updating the weights of a neural network to improve its performance on any given task`.
- Neural networks align with Arthur Samuel's original vision for machine learning, offering a versatile solution for a broad range of problems through weight optimization.
- The effectiveness of weight assignments is measured by the model's accuracy in making correct predictions, aligning with Samuel's framework for evaluating performance.
- In the context of image classification, the inputs are images, the weights are the neural network's parameters, and the model's output could classify images.

