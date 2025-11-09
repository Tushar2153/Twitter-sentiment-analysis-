# 🧠 Twitter Hate Speech Detection

This project detects hate speech in tweets by classifying them as racist/sexist (`label = 1`) or not (`label = 0`). It uses natural language processing (NLP) techniques and a Logistic Regression model trained on a labeled dataset of 31,962 tweets.

---

## 📁 Dataset

- **File**: `train_E60V31V.csv`
- **Size**: 31,962 tweets
- **Columns**:
  - `id`: Unique tweet ID
  - `label`: 1 = hate speech, 0 = non-hate speech
  - `tweet`: Raw tweet text

---

## 🧹 Preprocessing

- Removed Twitter handles (`@user`)
- Removed special characters, numbers, and punctuation
- Removed short words (≤ 3 characters)
- Tokenized tweets into words
- Applied stemming using PorterStemmer
- Reconstructed cleaned tweets

---

## 📊 Exploratory Data Analysis

### WordClouds
Visualized frequent words for:
- All tweets
- Non-hate tweets (`label = 0`)
- Hate tweets (`label = 1`)

### Hashtag Analysis
- Extracted hashtags from both classes
- Top 10 hashtags visualized using bar plots

#### Top Positive Hashtags
- `love`, `posit`, `smile`, `healthi`, `thank`, `fun`, `affirm`, `life`, `model`, `summer`

#### Top Negative Hashtags
- `trump`, `polit`, `allahsoil`, `liber`, `libtard`, `sjw`, `retweet`, `black`, `miamiâ`, `hate`

---

## 🔍 Feature Extraction

Used Bag-of-Words (BoW) with `CountVectorizer`:

```python
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(
    max_df=0.90,
    min_df=2,
    max_features=1000,
    stop_words='english'
)
bow = bow_vectorizer.fit_transform(df['clean_tweet'])
