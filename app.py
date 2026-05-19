from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
 
nltk.download('stopwords')
 
app = Flask(__name__)
 
# Load the model and vectorizer
model = pickle.load(open('train_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
 
port_stem = PorterStemmer()
 
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tweet = data['tweet']
 
    # Preprocess exactly like training
    processed = stemming(tweet)
 
    # Vectorize
    vectorized = vectorizer.transform([processed])
 
    # Predict
    prediction = model.predict(vectorized)
    result = int(prediction[0])
 
    label = 'Positive 😊' if result == 1 else 'Negative 😞'
    return jsonify({'prediction': result, 'label': label})
 
if __name__ == '__main__':
    app.run(debug=True)