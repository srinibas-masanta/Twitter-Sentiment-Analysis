# Twitter Sentiment Analysis using Machine Learning

Welcome to the **Twitter Sentiment Analysis Project**! This project uses machine learning techniques to analyze the sentiment of tweets—whether they convey positive or negative emotions. Our primary focus is to build a robust predictive model capable of classifying tweets based on their sentiment, empowering businesses, researchers, or developers to gauge the public's opinion on various topics. 

The dataset used for this project is the [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/datasets/kazanova/sentiment140/data) sourced from Kaggle. The dataset is extensive, containing both positive and negative tweets, making it a great resource for training machine learning models.

## Dataset

The **Sentiment140 dataset** contains 1.6 million tweets with labels:
- **0** = Negative Sentiment
- **4** = Positive Sentiment (later mapped to 1 for simplicity)

Each tweet is accompanied by several metadata fields (user ID, date, etc.), but for the purpose of sentiment analysis, we focus primarily on the tweet text and sentiment labels.

**Key Data Columns**:
- `target`: Sentiment label (0 = negative, 1 = positive)
- `text`: Tweet content

## Project Purpose

The goal of this project is to:
- **Analyze the sentiment of tweets**: Distinguish between positive and negative sentiments.
- **Build predictive models**: Use machine learning algorithms to classify tweets based on their content.
- **Gain insights**: Understand the strengths and limitations of different machine learning models applied to text classification.

## Key Features

- **Preprocessing Tweets**: A comprehensive text preprocessing pipeline, including removing special characters, links, and usernames, tokenizing words, and applying stemming techniques.
- **Model Building**: Two models were implemented and evaluated:
  1. **Logistic Regression**: A reliable and interpretable baseline model.
  2. **Naive Bayes**: A simple yet powerful algorithm for text classification.
- **Model Comparison**: Accuracy, precision, recall, and F1-scores were calculated for both models to determine the best-performing one.
- **Visual Insights**: Confusion matrix visualizations with both count and percentage values for clearer understanding.
- **ROC Curve Analysis**: ROC and AUC scores were plotted to assess the trade-off between sensitivity and specificity.
  
## Workflow

### 1. **Dataset Exploration**
   We started with the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140/data) consisting of 1.6 million tweets. The target labels are:
   - `0` for negative sentiment
   - `1` for positive sentiment

### 2. **Data Preprocessing**
   - **Cleaning the text**: Removed unwanted noise such as URLs, mentions, and special characters.
   - **Tokenization and Stemming**: Converted the cleaned tweets into tokens and applied stemming using the PorterStemmer.
   - **Vectorization**: Used TF-IDF (Term Frequency-Inverse Document Frequency) to convert the preprocessed tweets into numerical format suitable for model training.

### 3. **Model Training**
   Two machine learning models were implemented:
   - **Logistic Regression**: A linear model providing a strong baseline with accuracy scores of **79.65%** on the training data and **78.02%** on the test data.
   - **Naive Bayes**: Achieved **79.95%** accuracy on the training data and **76.30%** on the test data.

   These models were trained using **80%** of the data and tested on the remaining **20%** to evaluate their performance.

### 4. **Model Evaluation**
   - **Confusion Matrix**: Plotted confusion matrices to visualize the model's predictions. Each cell shows the counts and corresponding percentages of correct and incorrect predictions for positive and negative tweets.
   - **Classification Report**: Calculated key metrics like **precision**, **recall**, and **F1-score** for both positive and negative sentiment classes.
   - **ROC Curve**: Plotted the ROC curve and computed the AUC score to evaluate the trade-off between the true positive and false positive rates.

### 5. **Model Persistence**
   The trained models were saved using Python's **pickle** module for future use, allowing predictions on unseen data without retraining the models.

## Model Comparison and Key Insights

| Metric           | Logistic Regression | Naive Bayes  |
|------------------|---------------------|--------------|
| **Training Accuracy** | 79.65%              | 79.95%       |
| **Test Accuracy**     | 78.02%              | 76.30%       |
| **Precision (Pos)**   | 0.77                | 0.77         |
| **Recall (Pos)**      | 0.80                | 0.75         |

- **Logistic Regression** consistently provided better overall performance, making it a great choice for this problem. It achieved balanced precision and recall for both positive and negative tweets.
- **Naive Bayes** was slightly less accurate but offers a lightweight and interpretable alternative, particularly excelling in terms of recall for negative tweets.
- **Text preprocessing** plays a crucial role in improving model accuracy. Removing noise and stemming helped improve the predictions significantly.
- **TF-IDF Vectorization** effectively transformed text data into meaningful numeric representations, capturing the importance of words in the dataset.

## Future Enhancements
- Explore other machine learning algorithms such as **SVM** or deep learning models like **RNNs** for improved accuracy.
- Implement advanced NLP techniques such as **word embeddings** (Word2Vec, GloVe) to capture semantic meaning in tweets.
- Create a user-friendly web app using **Streamlit** or **Flask** to deploy the sentiment analysis model for real-time predictions.

## Conclusion

This project demonstrates how machine learning can be used to analyze the sentiment of tweets and how different models behave in text classification tasks. By implementing and evaluating Logistic Regression and Naive Bayes models, we were able to gain valuable insights into the effectiveness of each approach.

Future enhancements could involve using more advanced techniques such as **deep learning** or **transformer-based models** (e.g., BERT) to further boost classification accuracy.
