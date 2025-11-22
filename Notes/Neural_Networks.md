
# Neural Network
A **Neural Network** is a computational model inspired by the structure and functioning of the human brain. It is made up of interconnected units called **neurons** that work together to process information and learn patterns from data.

### **Simple Explanation**

A neural network takes some input, processes it through several layers of neurons, and produces an output. It learns by adjusting the strength (weights) of connections between neurons based on the errors it makes. Over time, this allows the network to recognize patterns, make predictions, or classify data.

### **Definition**
A **Neural Network** is a machine learning model that consists of interconnected processing elements (neurons) arranged in layers. It learns from examples by adjusting connection weights and is capable of recognizing patterns, approximating functions, and solving complex problems.

## **Key Features**

1. **Layers:** Input layer → Hidden layer(s) → Output layer
2. **Neurons:** Basic units that compute weighted sums and apply activation functions.
3. **Weights:** Values that determine the strength of connections.
4. **Learning:** Achieved by updating weights using a learning algorithm (e.g., backpropagation).
5. **Non-linearity:** Activation functions allow the network to learn complex patterns.

## **Why Neural Networks Are Useful**

* Can learn relationships that are difficult to express mathematically.
* Handle noisy, incomplete, or unstructured data.
* Widely used in image recognition, speech processing, natural language tasks, and prediction systems.

----

# Activation Functions
An Activation Function is a mathematical function used in a neural network to determine whether a neuron should be activated or not. It introduces non-linearity into the model, allowing the network to learn complex patterns instead of just simple linear relationships.

## **Purpose of Activation Functions**

1. **Introduce Non-Linearity:** Allows the network to learn complicated relationships.
2. **Control Output Range:** Keeps neuron outputs within a certain limit.
3. **Decide Neuron Activation:** Determines whether a neuron should “fire” or not.
4. **Help in Learning:** Makes backpropagation possible by providing differentiable functions.


## **Common Activation Functions**

1. **Sigmoid:** Output between 0 and 1, used for binary classification.
2. **ReLU (Rectified Linear Unit):** Most used; outputs zero for negative values and linear for positive values.
3. **Tanh:** Output between –1 and 1, centered around zero.
4. **Softmax:** Used in the output layer for multi-class classification.

## Some of the **Activation Functions**

### **1. Sigmoid Activation Function**

The Sigmoid function squashes input values into a range between **0 and 1**, making it useful for binary classification.

### **2. Tanh (Hyperbolic Tangent) Activation Function**

The Tanh function outputs values between **–1 and +1** and is zero-centered, allowing better gradient flow than Sigmoid.

### **3. ReLU (Rectified Linear Unit) Activation Function**

ReLU outputs **0 for negative values** and the **input itself for positive values**. It is simple and fast, making it widely used in deep networks.

### **4. Leaky ReLU Activation Function**

Leaky ReLU allows a **small, non-zero slope** for negative values, preventing neurons from dying (outputting only zero).

### **5. Softmax Activation Function**

Softmax converts a vector of numbers into a **probability distribution** where all outputs sum to 1, used for multi-class classification.

----

# **ANN vs RNN vs CNN**

## **1. ANN (Artificial Neural Network)**

An ANN is the basic form of a neural network. It is also known as a **Feed-Forward Network** because data moves only in one direction: from input to output. It may contain only input and output layers or may have hidden layers for learning complex patterns.

### **Advantages**

1. Stores information across the entire network.
2. Can work even when some input data is missing.
3. Provides fault tolerance due to distributed memory.
4. Learns and adapts from examples over time.

### **Disadvantages**

1. Requires high computational power and hardware.
2. Acts like a “black box,” making internal working difficult to interpret.
3. No fixed rules for choosing the network structure; often decided by trial-and-error.

---

## **2. RNN (Recurrent Neural Network)**

An RNN is designed for **sequential data** such as text, speech, or time-series. It contains **feedback/recurrent connections** that allow previous outputs to influence the current input. This helps the network retain memory of earlier steps.

### **Advantages**

1. Suitable for sequence-based and time-dependent data.
2. Remembers previous inputs using recurrent connections.
3. Uses shared weights, reducing the total number of parameters.

### **Disadvantages**

1. Suffers from the **vanishing/exploding gradient** problem.
2. Training is slow because processing is sequential.
3. Difficult to learn long-term dependencies without LSTM/GRU variants.

---

## **3. CNN (Convolutional Neural Network)**

A CNN is mainly used for **image and spatial data processing**. It uses **convolutional layers** to automatically learn features like edges, textures, and shapes from raw images.

### **Advantages**

1. Automatically extracts important features from images.
2. Requires fewer parameters due to weight sharing and local connectivity.
3. Achieves high accuracy in computer vision tasks such as object detection and image classification.

### **Disadvantages**

1. Needs large datasets and powerful hardware for training.
2. Not suitable for sequential tasks unless combined with other models.
3. Performance may reduce when inputs are rotated or distorted.
