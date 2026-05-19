# Twitter Sentiment Analysis 🚀

A Machine Learning based Sentiment Analysis project that classifies tweets as **Positive** or **Negative** using Natural Language Processing (NLP) techniques and Logistic Regression.

This project uses the famous **Sentiment140 dataset** containing 1.6 million tweets and performs complete text preprocessing, feature extraction using TF-IDF, model training, evaluation, and prediction.

---

# 📌 Project Overview

Sentiment Analysis is a Natural Language Processing (NLP) task used to determine whether a text expresses a **positive** or **negative** emotion.

In this project:

* Tweets are collected from the Sentiment140 dataset
* Text data is cleaned and preprocessed
* Stopwords are removed
* Words are stemmed using Porter Stemmer
* Tweets are converted into numerical vectors using TF-IDF
* A Logistic Regression model is trained
* The trained model predicts tweet sentiment

The final model is also saved using Pickle for future use.

---

# 🧠 Technologies Used

* Python
* NumPy
* Pandas
* NLTK
* Scikit-learn
* Logistic Regression
* TF-IDF Vectorization
* Pickle

---

# 📂 Dataset Used

Dataset: **Sentiment140**

The dataset contains:

* 1,600,000 tweets
* Positive and Negative sentiments

Target Labels:

* `0` → Negative Tweet
* `4` → Positive Tweet

In this project:

* `4` is converted to `1`

---

# ⚙️ Project Workflow

## 1️⃣ Importing Libraries

The required Python libraries are imported:

```python
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

These libraries help in:

* Data handling
* Text preprocessing
* Machine learning
* Model evaluation

---

# 2️⃣ Downloading and Extracting Dataset

The dataset is downloaded using Kaggle API:

```python
!kaggle datasets download -d kazanova/sentiment140
```

Then extracted using ZipFile:

```python
from zipfile import ZipFile
```

---

# 3️⃣ Loading the Dataset

The CSV dataset is loaded using Pandas:

```python
twitter_data = pd.read_csv(
    '/content/training.1600000.processed.noemoticon.csv',
    names=column_names,
    encoding='ISO-8859-1'
)
```

Columns:

* target
* ids
* date
* flag
* user
* text

---

# 4️⃣ Data Preprocessing

## Handling Labels

Positive sentiment label `4` is converted into `1`:

```python
twitter_data.replace({'target':{4:1}}, inplace=True)
```

So:

* `0` = Negative
* `1` = Positive

---

# 5️⃣ Text Cleaning and Stemming

A custom stemming function is created:

```python
def stemming(content):
```

The preprocessing steps include:

### ✔ Removing Special Characters

```python
re.sub('[^a-zA-Z]', ' ', content)
```

### ✔ Converting to Lowercase

```python
content.lower()
```

### ✔ Splitting Words

```python
content.split()
```

### ✔ Removing Stopwords

Common words like:

* the
* is
* and
* are

are removed using NLTK stopwords.

### ✔ Stemming

Words are reduced to their root form:

Examples:

* loving → love
* playing → play
* studies → studi

using:

```python
PorterStemmer()
```

---

# 6️⃣ Feature Extraction using TF-IDF

Text data cannot be directly used in Machine Learning models.

So tweets are converted into numerical vectors using:

```python
TfidfVectorizer()
```

TF-IDF helps identify important words in tweets while reducing the impact of common words.

---

# 7️⃣ Splitting Training and Testing Data

Dataset is divided into:

* Training Data → 80%
* Testing Data → 20%

```python
train_test_split()
```

This helps evaluate model performance on unseen data.

---

# 8️⃣ Model Training

The Machine Learning model used is:

## Logistic Regression

```python
model = LogisticRegression(max_iter=1000)
```

The model learns patterns from tweet text and predicts sentiment.

---

# 9️⃣ Model Evaluation

Accuracy is calculated for:

* Training Data
* Testing Data

Using:

```python
accuracy_score()
```

This measures how correctly the model predicts sentiments.

---

# 🔟 Saving the Model

The trained model is saved using Pickle:

```python
pickle.dump(model, open(filename, 'wb'))
```

Saved file:

```bash
train_model.sav
```

This allows reuse without retraining.

---

# 1️⃣1️⃣ Making Predictions

The saved model is loaded:

```python
loaded_model = pkl.load(open('/content/train_model.sav','rb'))
```

Then predictions are made on sample tweets.

Output:

* Positive Tweet
* Negative Tweet

---

# 📊 Model Performance

The project achieves high accuracy using Logistic Regression and TF-IDF vectorization on the Sentiment140 dataset.

Typical accuracy:

* Training Accuracy: ~80%
* Testing Accuracy: ~77-79%

(Actual accuracy may vary slightly.)

---

# 📁 Project Structure

```bash
Twitter-Sentiment-Analysis/
│
├── Twitter_sentiments.ipynb
├── train_model.sav
├── kaggle.json
├── README.md
└── dataset/
```

---

# ▶️ How to Run the Project

## Step 1: Clone Repository

```bash
git clone https://github.com/your-username/twitter-sentiment-analysis.git
```

## Step 2: Install Dependencies

```bash
pip install numpy pandas nltk scikit-learn
```

## Step 3: Run Notebook

Open:

```bash
Twitter_sentiments.ipynb
```

Run all cells sequentially.

---

# 💡 Future Improvements

Possible improvements:

* Add Neutral Sentiment class
* Use Deep Learning models (LSTM, GRU)
* Use BERT Transformer model
* Deploy using Flask or Streamlit
* Create real-time Twitter sentiment analyzer

---

# 📚 Learning Outcomes

Through this project, concepts learned include:

* NLP preprocessing
* Text cleaning
* Stopword removal
* Stemming
* TF-IDF Vectorization
* Logistic Regression
* Model Evaluation
* Model Serialization using Pickle

---

# 🤝 Contributing

Contributions are welcome.

You can:

* Improve preprocessing
* Try different ML models
* Add deployment support
* Optimize accuracy

---

# 📜 License

This project is open-source and available under the MIT License.

---

# 👨‍💻 Author

Developed by **Roshan Karnakar**

Machine Learning & NLP Enthusiast 🚀
