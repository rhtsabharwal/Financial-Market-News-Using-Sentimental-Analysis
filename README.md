# Financial-Market-News-Using-Sentimental-Analysis


# Financial Market News Sentiment Analysis

This project uses machine learning to **analyze the sentiment of financial news headlines** and predict whether the stock market will go up or down based on that sentiment.

We process news data using **TF-IDF vectorization**, and then train a **Logistic Regression** model to classify the sentiment as **positive (Label: 1)** or **negative (Label: 0)**.

---

## Project Overview

- Input: A dataset of daily financial news headlines (`NEWS.csv`)
- Goal: Predict whether the market sentiment is positive or negative for each day
- Technique: Natural Language Processing (NLP) with TF-IDF and Logistic Regression
- Output: A binary classification — Market sentiment: **Positive** or **Negative**

---

## Tools & Libraries Used

- Python 
- pandas, numpy
- scikit-learn (TF-IDF, Logistic Regression, train/test split)
- seaborn & matplotlib for visualization

---

##  How It Works

1. **Read the data** from a CSV file containing financial headlines and sentiment labels
2. **Combine all headlines** into a single string for each day
3. **Convert text to numerical features** using `TfidfVectorizer`
4. **Split the data** into training and testing sets
5. **Train a Logistic Regression model** on the training data
6. **Predict and evaluate** the model on the test data using accuracy and confusion matrix

---

##  Model Performance

- We use `accuracy_score`, `classification_report`, and a **confusion matrix** to evaluate the predictions
- Visual output includes a heatmap of the confusion matrix to see how well the model performs
