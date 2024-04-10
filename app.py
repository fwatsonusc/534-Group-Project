from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import openai

app = Flask(__name__)

# Initialize OpenAI API
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Initialize Vader Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def get_emotion(text):
    # Analyze sentiment using Vader
    sentiment_score = analyzer.polarity_scores(text)
    
    # Convert sentiment scores to emotions
    if sentiment_score['compound'] >= 0.05:
        emotion = 'positive'
    elif sentiment_score['compound'] <= -0.05:
        emotion = 'negative'
    else:
        emotion = 'neutral'
    
    return emotion

def generate_response(prompt):
    # Generate response from OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=100
    )
    return response.choices[0].text.strip()

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    user_input = data['user_input']
    emotion = get_emotion(user_input)
    prompt = f"You are feeling {emotion}. Please describe your IT issue:\n{user_input}\nResponse:"
    response = generate_response(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
