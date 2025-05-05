from flask import Flask, request, render_template
import joblib
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('model.pkl')  # Path to your saved model
vectorizer = joblib.load('vectorizer.pkl')  # Path to your saved vectorizer

# Define the clean_text function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(words)

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Load the homepage template

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input news article from the form
        news = request.form['news']

        # Preprocess the text using the clean_text function
        cleaned = clean_text(news)

        # Transform the input text using the vectorizer
        vector = vectorizer.transform([cleaned])

        # Make a prediction using the model
        pred = model.predict(vector)

        # Map prediction to label
        label = 'REAL' if pred[0] == 1 else 'FAKE'

        # Render the homepage with the prediction result
        return render_template('index.html', prediction=label)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)