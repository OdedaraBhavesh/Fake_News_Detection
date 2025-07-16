from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

ps = PorterStemmer()


def preprocess_text(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [ps.stem(word)
               for word in content if word not in stopwords.words('english')]
    return " ".join(content)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_news = request.form['news']
    processed_news = preprocess_text(input_news)
    vector_input = vectorizer.transform([processed_news])
    prediction = model.predict(vector_input)[0]
    result = 'Fake News ❌' if prediction == 1 else 'Real News ✅'
    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
