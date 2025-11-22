# **Supervised Learning**
---

# **1. Introduction to Supervised Learning**

Supervised Learning is a machine learning approach where the model is trained using **labeled data**, meaning each input has a corresponding correct output.
The goal is to learn a **mapping function** from inputs (X) to outputs (Y).

### **Definition (Exam):**

Supervised Learning is a learning approach where a model learns from labeled training data to predict outputs for unseen inputs.

### **Examples:**

* Email spam classification
* Sentiment analysis
* House price prediction
* Disease detection

---

# **2. Learning a Class from Examples (Binary Classification)**

This refers to learning a function that assigns data into **two classes**, e.g., Yes/No, Spam/Not Spam.

### Steps:

1. Collect labeled examples
2. Choose a model (Decision Tree, Naive Bayes, Logistic Regression, etc.)
3. Train the model using input–output pairs
4. Evaluate performance using accuracy, confusion matrix, etc.

### Output:

* Class label (e.g., 0 or 1)

---

# **3. Learning Multiple Classes (Multiclass Classification)**

Here, the target variable has **more than two classes**, e.g., classifying images of animals (Cat, Dog, Horse).

### Approaches:

1. **One-vs-One (OvO)**
2. **One-vs-Rest (OvR)**
3. Algorithms that naturally support multiclass:

   * Decision Trees
   * Random Forest
   * Neural Networks
   * k-NN

### Output:

* One of *k* possible classes.

---

# **4. Model Selection and Generalization**

Choosing the model that performs best on unseen data.

### **Key Concepts:**

* **Bias-Variance Tradeoff**
* **Model Complexity**
* **Underfitting and Overfitting**

### **Underfitting:**

Model is too simple → poor performance on both training and testing data.

### **Overfitting:**

Model memorizes training data → good training accuracy, poor test accuracy.

---

# **5. Training, Validation, and Testing** (from Module 1)

### **Training Set:**

Used to train the model by adjusting parameters.

### **Validation Set:**

Used to tune hyperparameters and select best model.

### **Test Set:**

Used for final performance evaluation.

---

# **6. Types of Error Calculation**

* **Training Error:** Error on training data
* **Validation Error:** Used for model selection
* **Test Error:** True generalization error

**Metrics:**

* Accuracy
* Precision/Recall
* F1 Score
* Mean Squared Error (Regression)
* MAE, RMSE

---

# **7. Linear Regression**

A supervised learning method for **predicting continuous values**.
It finds the best-fit line:

$$
y = mx + c
$$

### **Key Concepts:**

* Cost function: Mean Squared Error
* Gradient Descent for optimization
* Feature Selection (removing irrelevant features)

### **Feature Selection Methods:**

* Filter methods (Correlation, Chi-square)
* Wrapper methods (Forward/Backward selection)
* Embedded methods (Lasso, Ridge)

---

# **8. Bayesian Learning**

Bayesian learning uses **probability** to predict the class of an input using Bayes’ theorem:

$$
P(H|D) = \frac{P(D|H) \cdot P(H)}{P(D)}
$$

### **Example Algorithm:**

* **Naive Bayes Classifier**

  * Assumes features are independent
  * Works well for text classification

---

# **9. Decision Tree Learning**

A decision tree splits data based on feature values.

### **Splitting Criteria:**

* **Gini Index**
* **Entropy / Information Gain**
* **Gain Ratio**

### **Advantages:**

* Easy to interpret
* Handles categorical and numerical data

### **Disadvantages:**

* Prone to overfitting

---

# **10. Classification Tree**

Used when the **output is categorical**.

### **Examples:**

* Yes/No decisions
* Multi-class classification

### **Node Impurity Measures:**

* Gini impurity
* Entropy

---

# **11. Regression Tree**

Used when the **output is continuous**.

### **Split Criteria:**

* Mean Squared Error
* Reduction in Variance

### **Leaf Node:**

* Holds a numeric value (average of samples)

---

# **12. Multivariate Methods for Learning**

Used when **multiple input variables** contribute to prediction.

### **(a) Multivariate Classification**

Predicting class labels using multiple features.
Algorithms include:

* Multivariate Logistic Regression
* Multivariate Decision Trees
* SVM
* Neural Networks

### **(b) Multivariate Regression**

Predicting multiple continuous outputs.
Example: predicting height and weight simultaneously.

---

# **13. Summary – Supervised Learning Workflow**

1. Collect labeled data
2. Preprocess and select features
3. Split into training/validation/testing
4. Select model
5. Train model
6. Tune hyperparameters
7. Evaluate using test data
8. Deploy the model

