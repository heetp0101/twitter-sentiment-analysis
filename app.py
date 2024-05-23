from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model
with open('twitter_sentiment.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the tweet text from the form
        tweet = request.form['tweet']
        
        # Process the tweet as needed (this function should be defined based on your model requirements)
        processed_tweet = process_tweet(tweet)
        
        # Make prediction
        prediction = model.predict([processed_tweet])
        
        # Return the result as JSON
        return jsonify({'prediction': prediction[0]})

def process_tweet(tweet):
    # Implement any preprocessing steps here (e.g., tokenization, vectorization)
    return tweet

if __name__ == "__main__":
    app.run(debug=True)
