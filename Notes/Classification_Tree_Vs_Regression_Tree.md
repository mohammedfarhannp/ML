# **Classification Tree vs Regression Tree (10-Mark Answer)**

Decision Trees are widely used supervised learning models that split data into branches based on input features. They are mainly of two types: **Classification Trees** and **Regression Trees**. Although both use a tree-like structure, they differ in purpose, splitting criteria, and the type of output produced.

---

## **1. Classification Trees**

A **Classification Tree** is used when the **target variable is categorical**. The goal is to assign an input instance to one of several predefined classes.

### **How It Works**

* The tree splits data based on feature values to increase the *purity* of the resulting nodes.
* At each split, the algorithm chooses a feature that best separates the classes of data.

### **Splitting Criteria**

Common measures of node impurity include:

1. **Gini Index**
2. **Entropy / Information Gain**
3. **Misclassification Error**

### **Output**

* The leaf nodes represent **class labels**.
* Example: "Spam" vs "Not Spam", "Disease Present" vs "No Disease".

### **Advantages**

* Simple and easy to interpret.
* Works well with categorical data.
* No need for feature scaling.

### **Disadvantages**

* Can overfit on training data.
* Less stable—small changes in data may change the tree structure.

---

## **2. Regression Trees**

A **Regression Tree** is used when the **target variable is continuous**. The goal is to predict a numerical value based on input features.

### **How It Works**

* The tree splits data into regions that minimize the **variance** of the target variable within each region.
* Every leaf node contains a **numeric value**, usually the mean of training samples in that region.

### **Splitting Criteria**

Regression Trees commonly use:

1. **Mean Squared Error (MSE)**
2. **Mean Absolute Error (MAE)**
3. **Reduction in Variance**

### **Output**

* Leaf nodes represent **continuous predicted values**.
* Example: Predicting house prices, temperature, or stock price.

### **Advantages**

* Handles nonlinear relationships effectively.
* Simple to visualize and interpret compared to other regression models.

### **Disadvantages**

* Can be unstable and prone to overfitting.
* Predictions may be piecewise constant, leading to lower accuracy for smooth relationships.

---

## **3. Key Differences (Point-wise)**

| Aspect                  | Classification Tree       | Regression Tree               |
| ----------------------- | ------------------------- | ----------------------------- |
| **Target Variable**     | Categorical               | Continuous                    |
| **Output**              | Class labels              | Numeric values                |
| **Splitting Criteria**  | Gini, Entropy             | MSE, MAE                      |
| **Node Purity Measure** | Class purity              | Variance reduction            |
| **Decision at Leaves**  | Most frequent class       | Mean value of samples         |
| **Use Cases**           | Spam detection, diagnosis | Price prediction, forecasting |

---

## **4. Applications**

* **Classification Trees:** Medical diagnosis, fraud detection, text classification, customer churn prediction.
* **Regression Trees:** Real estate valuation, weather forecasting, financial predictions, sales forecasting.

---

# **Conclusion**
Classification Trees and Regression Trees share a common structure but differ fundamentally in their target outputs and splitting criteria. Classification Trees categorize data into classes, while Regression Trees predict continuous values. Understanding these differences is essential for choosing the correct model based on the type of data and problem requirements.
