import joblib
from flask import Flask, render_template, request

# Load the trained model
model = joblib.load('fake_news_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        prediction = model.predict([news_text])
        
        # Show result
        if prediction == 1:
            result = "Fake News"
        else:
            result = "Real News"
        return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
