Here is a `README.md` file for this project, based on the provided document.

-----

# Twitter Hate Speech Detection

This project detects hate speech in tweets by classifying them as racist/sexist or not. The model is built using a Logistic Regression classifier and a Bag-of-Words approach for feature extraction.

## 📜 Project Objective

[cite\_start]The main goal is to identify tweets that contain racist or sexist sentiment[cite: 2, 3].
Given a dataset of tweets, the model predicts a label:

  * [cite\_start]**Label '1'**: The tweet is racist or sexist[cite: 4].
  * [cite\_start]**Label '0'**: The tweet is not racist or sexist[cite: 4].

## 💾 Dataset

[cite\_start]The model is trained on a labeled dataset containing **31,962 tweets**[cite: 5]. [cite\_start]The data is provided in a CSV file (`train_E60V31V.csv`) with the following columns[cite: 6]:

  * `id`: A unique identifier for the tweet.
  * `label`: The sentiment label (0 or 1).
  * `tweet`: The raw text of the tweet.

## 🛠️ Project Pipeline

The project follows these key steps:

### 1\. Data Preprocessing

Before analysis and modeling, the raw tweet text is cleaned using the following steps:

  * [cite\_start]**Remove Twitter Handles**: All handles (e.g., `@user`) are removed[cite: 57, 58].
  * [cite\_start]**Remove Special Characters**: Punctuation, numbers, and special characters are removed, preserving only letters and hashtags (`#`)[cite: 84, 85].
  * [cite\_start]**Remove Short Words**: Words that are very short (e.g., 2-3 letters) are removed[cite: 109].
  * [cite\_start]**Tokenization**: Each cleaned tweet is split into individual words (tokens)[cite: 111].
  * [cite\_start]**Stemming**: Words are reduced to their root form using `nltk.PorterStemmer` (e.g., "dysfunctional" becomes "dysfunct")[cite: 111].
  * [cite\_start]**Rejoin**: The stemmed tokens are re-joined into a single string for feature extraction[cite: 111].

### 2\. Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset:

  * **Word Clouds**: Visualizations were created to show the most frequent words in:
      * [cite\_start]All tweets combined[cite: 115].
      * [cite\_start]Non-hate tweets (label 0)[cite: 178].
      * [cite\_start]Hate-speech tweets (label 1)[cite: 239].
  * [cite\_start]**Hashtag Analysis**: Hashtags were extracted from both positive and negative tweets[cite: 266, 276, 278].
      * [cite\_start]**Top Positive Hashtags**: `love`, `posit`, `smile`, `healthi`, `thank`[cite: 308, 309, 310, 311, 312].
      * [cite\_start]**Top Negative Hashtags**: `trump`, `polit`, `allahsoil`, `liber`, `libtard`[cite: 332, 333, 334, 335, 336].

### 3\. Model Training

  * **Feature Extraction**: The cleaned tweet text was converted into numerical features using `CountVectorizer` (Bag-of-Words). [cite\_start]The vectorizer was configured with `max_features=1000`, `max_df=0.90`, and `min_df=2`[cite: 344].
  * [cite\_start]**Train-Test Split**: The dataset was split into training and testing sets[cite: 348].
  * [cite\_start]**Modeling**: A **Logistic Regression** model was trained on the data[cite: 352, 355].

## 📊 Results

The model's performance was evaluated on the test set using F1-score and accuracy.

**Initial Model Performance:**

  * [cite\_start]**Accuracy:** 0.948 [cite: 366]
  * [cite\_start]**F1-Score:** 0.508 [cite: 364]

**Performance after Tuning (adjusting probability threshold):**

  * [cite\_start]**Accuracy:** 0.944 [cite: 377]
  * [cite\_start]**F1-Score:** 0.561 [cite: 375]

## 📦 Dependencies

This project uses the following Python libraries:

  * [cite\_start]`pandas` [cite: 8]
  * [cite\_start]`numpy` [cite: 10]
  * [cite\_start]`matplotlib` [cite: 11]
  * [cite\_start]`seaborn` [cite: 11]
  * [cite\_start]`nltk` [cite: 14]
  * [cite\_start]`re` [cite: 12]
  * [cite\_start]`wordcloud` [cite: 117]
  * [cite\_start]`scikit-learn` (sklearn) [cite: 344, 348, 352]

## 🚀 How to Use

1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn nltk wordcloud scikit-learn
    ```
3.  Make sure the `train_E60V31V.csv` file is in the project's root directory.
4.  Run the Python script or Jupyter Notebook.
