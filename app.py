from flask import Flask, request, jsonify, render_template # type: ignore
import joblib # type: ignore

# Load the model and vectorizer
model = joblib.load('symptom_checker_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Text cleaning function
def clean_text(text):
    import re
    import nltk # type: ignore
    from nltk.corpus import stopwords # type: ignore
    from nltk.tokenize import word_tokenize # type: ignore

    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return ' '.join(tokens)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the request
    user_input = request.json['symptoms']
    
    # Clean and preprocess the input
    cleaned_input = clean_text(user_input)
    
    # Vectorize the input
    input_vec = vectorizer.transform([cleaned_input])
    
    # Make a prediction
    prediction = model.predict(input_vec)
    
    # Return the prediction as JSON
    return jsonify({'predicted_disease': prediction[0]})

# Define the home route to serve the frontend
@app.route('/')
def home():
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)