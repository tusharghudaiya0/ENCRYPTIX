from flask import Flask, request, render_template
import joblib
import string
from nltk.corpus import stopwords
import nltk
import os

nltk.download('stopwords')

app = Flask(__name__)

MODEL_PATH = r'C:\Users\User\OneDrive\Desktop\Tushar\Encryptix Projects\SMS SPAM COLLECTION\classifier_model.pkl'
VECTORIZER_PATH = r'C:\Users\User\OneDrive\Desktop\Tushar\Encryptix Projects\SMS SPAM COLLECTION\Tfid_vectorized.pkl'


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"The model file '{MODEL_PATH}' was not found.")

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"The vectorizer file '{VECTORIZER_PATH}' was not found.")

model = joblib.load(MODEL_PATH)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

def text_process(mess):
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        cleaned_message = text_process(message)
        data = [cleaned_message]
        transformed_data = tfidf_vectorizer.transform(data)
        prediction = model.predict(transformed_data)
        result = 'Spam' if prediction[0] == 1 else 'Not Spam'
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
