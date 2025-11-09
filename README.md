# Twitter Hate Speech Detection

This project uses Natural Language Processing (NLP) and a machine learning model to detect hate speech in tweets. The primary objective is to classify tweets as racist/sexist or not. The model is built using `scikit-learn` and `nltk` and relies on a Bag-of-Words (BoW) feature extraction method.

## 📜 Project Objective

The goal of this task is to identify tweets containing hate speech. For this project, a tweet is considered hate speech if it has **racist or sexist sentiment**.

The task is to build a classification model that, given a set of tweets, will predict one of two labels:
* **Label '1'**: The tweet is racist/sexist.
* **Label '0'**: The tweet is not racist/sexist.

## 💾 Dataset

The model was trained on a labeled dataset containing **31,962 tweets**.

* **File**: `train_E60V31V.csv`
* **Format**: The CSV file contains three columns:
    * `id`: `int64` - A unique identifier for each tweet.
    * `label`: `int64` - The sentiment label (0 or 1).
    * `tweet`: `object` - The raw text of the tweet.

---

## 🛠️ Project Pipeline

The project follows a standard NLP workflow: Data Preprocessing, Exploratory Data Analysis (EDA), Feature Extraction, and Model Training.

### 1. Data Preprocessing

The raw tweet text (`tweet` column) was cleaned to create a new `clean_tweet` column. This involved several steps:

1.  **Remove Twitter Handles**: Removed all Twitter handles (e.g., `@user`) using a custom `remove_pattern` function with the regex `@[\w]*`.
    ```python
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for word in r:
            input_txt = re.sub(word, "", input_txt)
        return input_txt
    
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
    ```

2.  **Remove Special Characters**: Removed all special characters, numbers, and punctuation, *except* for the hashtag symbol (`#`).
    ```python
    df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
    ```

3.  **Remove Short Words**: Words with a length of 3 or less were removed (e.g., "so", "is").

4.  **Tokenization**: Each cleaned tweet was split into a list of individual words (tokens).
    ```python
    tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())
    ```

5.  **Stemming**: Tokens were reduced to their root form using the `PorterStemmer` from `nltk`. (e.g., "dysfunctional" becomes "dysfunct").
    ```python
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
    ```

6.  **Rejoin Tokens**: The list of stemmed tokens was joined back into a single string for each tweet.

### 2. Exploratory Data Analysis (EDA)

EDA was performed to find insights in the cleaned data:

* **Word Clouds**: Word clouds were generated to visualize the most frequent words in:
    1.  All tweets combined.
    2.  Non-hate tweets (label 0).
    3.  Hate-speech tweets (label 1).
    *Words like `love`, `happi`, and `bihday` were common in positive tweets, while `trump`, `white`, `black`, `libtard`, and `allahsoil` were prominent in negative tweets.*

* **Hashtag Analysis**: Hashtags were extracted from all tweets using the regex `r"#(\w+)"`.
    * The top 10 most frequent hashtags from **non-hate tweets** (label 0) were plotted. Top 3: `#love`, `#posit`, `#smile`.
    * The top 10 most frequent hashtags from **hate-speech tweets** (label 1) were plotted. Top 3: `#trump`, `#polit`, `#allahsoil`.

### 3. Feature Extraction

The cleaned, stemmed text was converted into a numerical format suitable for modeling.

* **Technique**: Bag-of-Words (BoW)
* **Implementation**: `sklearn.feature_extraction.text.CountVectorizer`.
* **Parameters**:
    * `max_df=0.90` (Ignore terms that appear in > 90% of documents).
    * `min_df=2` (Ignore terms that appear in < 2 documents).
    * `max_features=1000` (Use only the top 1000 most frequent terms).
    * `stop_words='english'` (Remove common English stop words).

### 4. Model Building & Evaluation

1.  **Train-Test Split**: The BoW features (`bow`) and the target (`df['label']`) were split into training and testing sets using `train_test_split`.

2.  **Model Selection**: A **Logistic Regression** model was chosen for classification.
    ```python
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(x_train, y_train)
    ```

3.  **Evaluation**: The model was evaluated using `f1_score` (crucial for imbalanced datasets) and `accuracy_score`.

---

## 📊 Results

The model's performance was evaluated in two stages:

**1. Default Model (Using `.predict()`):**
* **F1-Score**: 0.508
* **Accuracy**: 0.948

**2. Tuned Model (Adjusting Probability Threshold):**
By adjusting the prediction threshold for the positive class, the F1-score was improved.
* **F1-Score**: 0.561
* **Accuracy**: 0.944

This shows that tuning the probability threshold for the positive class (label 1) can significantly improve the F1-score, which is a better measure of success for this type of imbalanced classification problem.

---

## 📦 Core Dependencies

This project relies on the following Python libraries:

* **Data Manipulation**: `pandas`, `numpy`
* **Data Visualization**: `matplotlib`, `seaborn`
* **Text Processing**: `re`, `string`, `nltk`
* **NLP/ML**: `wordcloud`, `scikit-learn` (sklearn)
* **Utilities**: `warnings`
