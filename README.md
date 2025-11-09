# Twitter Hate Speech Detection

This project uses Natural Language Processing (NLP) and a machine learning model to detect hate speech in tweets. The primary objective is to classify tweets as racist/sexist or not. The model is built using `scikit-learn` and `nltk` and relies on a Bag-of-Words (BoW) feature extraction method.

## 📜 Project Objective

[cite_start]The goal of this task is to identify tweets containing hate speech[cite: 2]. [cite_start]For this project, a tweet is considered hate speech if it has **racist or sexist sentiment**[cite: 2, 3].

The task is to build a classification model that, given a set of tweets, will predict one of two labels:
* [cite_start]**Label '1'**: The tweet is racist/sexist[cite: 4].
* [cite_start]**Label '0'**: The tweet is not racist/sexist[cite: 4].

## 💾 Dataset

[cite_start]The model was trained on a labeled dataset containing **31,962 tweets**[cite: 5, 20].

* [cite_start]**File**: `train_E60V31V.csv` [cite: 18]
* [cite_start]**Format**: The CSV file contains three columns[cite: 20]:
    * [cite_start]`id`: `int64` - A unique identifier for each tweet[cite: 22, 24].
    * [cite_start]`label`: `int64` - The sentiment label (0 or 1)[cite: 26, 27].
    * [cite_start]`tweet`: `object` - The raw text of the tweet[cite: 29, 31].

---

## 🛠️ Project Pipeline

The project follows a standard NLP workflow: Data Preprocessing, Exploratory Data Analysis (EDA), Feature Extraction, and Model Training.

### 1. Data Preprocessing

The raw tweet text (`tweet` column) was cleaned to create a new `clean_tweet` column. This involved several steps:

1.  [cite_start]**Remove Twitter Handles**: Removed all Twitter handles (e.g., `@user`) using a custom `remove_pattern` function with the regex `@[\w]*`[cite: 36, 58].
    ```python
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for word in r:
            input_txt = re.sub(word, "", input_txt)
        return input_txt
    
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
    ```

2.  [cite_start]**Remove Special Characters**: Removed all special characters, numbers, and punctuation, *except* for the hashtag symbol (`#`)[cite: 84].
    ```python
    df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
    ```
    [cite_start]*(Note: The PDF uses `""`[cite: 85], but ` " "` is generally preferred to avoid merging words.)*

3.  [cite_start]**Remove Short Words**: Words with a length of 3 or less were removed (e.g., "so", "is")[cite: 109, 110].

4.  [cite_start]**Tokenization**: Each cleaned tweet was split into a list of individual words (tokens)[cite: 111].
    ```python
    tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())
    ```

5.  [cite_start]**Stemming**: Tokens were reduced to their root form using the `PorterStemmer` from `nltk`[cite: 111]. (e.g., "dysfunctional" becomes "dysfunct").
    ```python
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
    ```

6.  [cite_start]**Rejoin Tokens**: The list of stemmed tokens was joined back into a single string for each tweet[cite: 111].

### 2. Exploratory Data Analysis (EDA)

EDA was performed to find insights in the cleaned data:

* **Word Clouds**: Word clouds were generated to visualize the most frequent words in:
    1.  [cite_start]All tweets combined[cite: 115].
    2.  [cite_start]Non-hate tweets (label 0)[cite: 178].
    3.  [cite_start]Hate-speech tweets (label 1)[cite: 239].
    [cite_start]*Words like `love`, `happi`, and `bihday` were common in positive tweets[cite: 133, 139, 147], while `trump`, `white`, `black`, `libtard`, and `allahsoil` were prominent in negative tweets (Page 6).*

* [cite_start]**Hashtag Analysis**: Hashtags were extracted from all tweets using the regex `r"#(\w+)"`[cite: 266, 269].
    * The top 10 most frequent hashtags from **non-hate tweets** (label 0) were plotted. [cite_start]Top 3: `#love`, `#posit`, `#smile` [cite: 297, 308-310].
    * The top 10 most frequent hashtags from **hate-speech tweets** (label 1) were plotted. [cite_start]Top 3: `#trump`, `#polit`, `#allahsoil` [cite: 324, 332-334].

### 3. Feature Extraction

The cleaned, stemmed text was converted into a numerical format suitable for modeling.

* **Technique**: Bag-of-Words (BoW)
* [cite_start]**Implementation**: `sklearn.feature_extraction.text.CountVectorizer`[cite: 344].
* **Parameters**:
    * [cite_start]`max_df=0.90` (Ignore terms that appear in > 90% of documents)[cite: 344].
    * [cite_start]`min_df=2` (Ignore terms that appear in < 2 documents)[cite: 344].
    * [cite_start]`max_features=1000` (Use only the top 1000 most frequent terms)[cite: 344].
    * [cite_start]`stop_words='english'` (Remove common English stop words)[cite: 344].

### 4. Model Building & Evaluation

1.  [cite_start]**Train-Test Split**: The BoW features (`bow`) and the target (`df['label']`) were split into training and testing sets using `train_test_split` with `random_state=4` (partially visible)[cite: 348, 349].

2.  [cite_start]**Model Selection**: A **Logistic Regression** model was chosen for classification[cite: 352, 356].
    ```python
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(x_train, y_train)
    ```

3.  [cite_start]**Evaluation**: The model was evaluated using `f1_score` (crucial for imbalanced datasets) and `accuracy_score`[cite: 353].

---

## 📊 Results

The model's performance was evaluated in two stages:

**1. Default Model (Using `.predict()`):**
* [cite_start]**F1-Score**: 0.508 [cite: 364]
* [cite_start]**Accuracy**: 0.948 [cite: 366]

**2. Tuned Model (Adjusting Probability Threshold):**
[cite_start]By adjusting the prediction threshold (the document shows `pred_prob [:,1] >= 8.3`, which is likely a typo for a value like `0.3` or `0.5`)[cite: 371], the F1-score was improved.
* [cite_start]**F1-Score**: 0.561 [cite: 374]
* [cite_start]**Accuracy**: 0.944 [cite: 376]

This shows that tuning the probability threshold for the positive class (label 1) can significantly improve the F1-score, which is a better measure of success for this type of imbalanced classification problem.

---

## 📦 Core Dependencies

This project relies on the following Python libraries:

* [cite_start]**Data Manipulation**: `pandas` [cite: 8][cite_start], `numpy` [cite: 10]
* [cite_start]**Data Visualization**: `matplotlib` [cite: 11][cite_start], `seaborn` [cite: 11]
* [cite_start]**Text Processing**: `re` [cite: 12][cite_start], `string` [cite: 13][cite_start], `nltk` [cite: 14]
* [cite_start]**NLP/ML**: `wordcloud` [cite: 117][cite_start], `scikit-learn` (sklearn) [cite: 344, 348, 352, 353]
* [cite_start]**Utilities**: `warnings` [cite: 15]
