# Clickbait-Spoiling-Multiclass-Classification
A final group project completed under NLP Course

# ANLP Final Project: Clickbait Spoiling Subtask 1 - Multiclass Classification

### University of Potsdam | MSc Cognitive Systems | BM1 “Advanced Natural Language Processing” | Winter 2023/2024

## Overview

This project was undertaken as part of the **Advanced Natural Language Processing** course. The goal was to tackle **Subtask 1** of the **SemEval 2023 Clickbait Spoiling Challenge**, focusing on **Multiclass Classification**. Specifically, the task involved classifying clickbait posts into one of three spoiler types: **phrase**, **passage**, or **multipart**.

We employed deep learning models, particularly **Long Short-Term Memory (LSTM)** networks, to achieve the classification. We also explored various natural language processing (NLP) techniques to preprocess the dataset, optimize the model, and evaluate the results.

---

## Project Structure

- `deneme6tensor.py`: TensorFlow implementation for model training and evaluation.

---

## Dataset

We used the **Webis Clickbait Spoiling Corpus 2022**, which contains around 5,000 posts. Each post is tagged with spoilers and is split into training, validation, and testing datasets (64/16/20 ratio). The dataset was processed using various preprocessing techniques to enhance model performance.

Sample dataset entry:
```json
{
  "uuid": "<UUID>",
  "postText": "<clickbait post>",
  "spoilerType": "<phrase | passage | multipart>"
}
```

---

## Approach

1. **Preprocessing**:
   - Tokenization of text data.
   - Removal of stop words and punctuation.
   - Conversion to lowercase.
   - Vectorization using **TF-IDF**.

2. **Model**:
   - **LSTM** networks with embedding layers for effective sequence learning.
   - **Bidirectional LSTMs** to capture context in both forward and backward directions.
   - Softmax activation for multiclass classification.

3. **Hyperparameter Tuning**:
   - Learning rate, batch size, and the number of layers were optimized for better performance.

4. **Evaluation**:
   - Accuracy, precision, recall, and F1-score were used as key metrics.
   - Confusion matrix for a detailed analysis of the model's predictions.

---

## Results

The trained model achieved the following performance on the test dataset:

| Metric      | Value    |
|-------------|----------|
| **Accuracy** | 0.437500 |
| **F1 Score** | 0.202899 |
| **Precision** | 0.812500 |
| **Recall**   | 0.333333 |

These metrics show that while the model had a high precision, its recall was relatively low, resulting in a moderate accuracy and F1 score. This indicates that the model was better at identifying spoilers correctly (high precision) but struggled to recall all relevant spoilers.

---

## Tools and Libraries

The following Python libraries were used in the implementation:
- **TensorFlow** and **Keras** for building and training the LSTM models.
- **Numpy** and **Pandas** for data manipulation.
- **Matplotlib** for visualizing the results.
- **Scikit-learn** for evaluation metrics and preprocessing.

---

## Conclusion

This project successfully implemented a spoiler classification system using LSTMs. We explored various preprocessing techniques and hyperparameter settings to improve the model's ability to classify clickbait posts into the correct spoiler type. While the performance metrics are promising, further optimization and experimentation could enhance the model’s predictive capabilities, particularly in improving the recall and F1 score.

---


## Acknowledgments

This project was completed as a group effort, with contributions in model design, evaluation, and implementation from all team members. We also acknowledge the guidance and support provided by our instructors during the course.
