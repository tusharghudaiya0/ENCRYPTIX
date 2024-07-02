from flask import Flask, request, render_template
import joblib
from text_cleaner import TextCleaner 

app = Flask(__name__)

model = joblib.load('movie_genre_predictor.pkl')

cleaner = TextCleaner()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    movie_description = request.form['description']
    
    cleaned_description = cleaner.cleaning_data(movie_description)
    
    prediction = model.predict([cleaned_description])[0]
    
    return render_template('index.html', prediction=prediction, description=movie_description)

if __name__ == '__main__':
    app.run(debug=True)
