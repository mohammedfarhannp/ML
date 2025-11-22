# **Module 1: Introduction to Machine Learning**

## **1. Concept of Learning Task**

A learning task in machine learning refers to the specific objective the model must achieve by analyzing data and identifying useful patterns. It involves understanding the relationship between inputs and outputs so that the system can improve its performance over time through experience.

## **2. Inductive Learning and Hypothesis Space**

Inductive learning is a process where the model generalizes patterns from observed examples and uses them to make predictions on unseen data. The hypothesis space represents the complete set of all possible models or functions the algorithm can choose from while searching for the best generalization.

## **3. Types of Machine Learning Approaches**

Machine learning approaches include supervised learning, unsupervised learning, and reinforcement learning, each differing in how data is labeled and how the model receives feedback. These approaches help algorithms solve a wide range of tasks such as classification, clustering, and decision-making.

## **4. Applications of Machine Learning**

Machine learning applications include areas like image recognition, natural language processing, fraud detection, medical diagnosis, and autonomous systems. These applications rely on data-driven decision-making to improve accuracy and efficiency in real-world tasks.

## **5. Supervised Learning**

Supervised learning is a method where the model learns from labeled data containing clear input-output pairs. It builds a predictive mapping by minimizing errors between predicted and actual values during training.

## **6. Unsupervised Learning**

Unsupervised learning deals with unlabeled data, allowing the model to uncover hidden structures or patterns without explicit guidance. It is widely used in clustering, dimensionality reduction, and anomaly detection.

## **7. Reinforcement Learning**

Reinforcement learning is a learning paradigm where an agent interacts with an environment and learns by receiving rewards or penalties based on its actions. This method helps the agent optimize long-term performance through trial and error.

## **8. Training, Validation, and Testing**

* **Training:** The model learns patterns from a labeled dataset by adjusting its internal parameters to reduce prediction errors.
* **Validation:** This dataset is used to tune hyperparameters and check whether the model is learning in a balanced way without overfitting.
* **Testing:** A completely unseen dataset used to evaluate the final model’s real-world performance and generalization ability.

## **9. Overfitting and Underfitting**

Overfitting occurs when a model learns noise or unnecessary detail from training data, reducing its ability to generalize to new inputs. Underfitting happens when the model is too simple to capture important patterns, resulting in poor performance on both training and test data.

## **10. Error Calculation Methods**

* **Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual values, making larger errors more significant.
* **Mean Absolute Error (MAE):** Computes the average absolute difference between predictions and true values, treating all errors equally.
* **Classification Error Rate:** Calculates the proportion of incorrect predictions to total predictions in classification tasks.
* **Cross-Entropy Loss:** Evaluates prediction confidence for classification tasks by comparing predicted probabilities with true labels.
* **Root Mean Squared Error (RMSE):** Provides the square root of MSE, making the error measure interpretable in original units.
