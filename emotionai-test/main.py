import os
from openai import OpenAI
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Vader Dictionary
nltk.download('vader_lexicon')

# Initialize Vader
sia = SentimentIntensityAnalyzer()

client = OpenAI(
    api_key="",  # Make sure to keep your API key secure
)

while True:
    user_input = input("Input: ")
    if user_input.lower() == "Exit":
        print("The chat is over。")
        break

    # Perform sentiment analysis
    sentiment_score = sia.polarity_scores(user_input)
    sentiment = 'Positive' if sentiment_score['compound'] > 0 else 'Negative' if sentiment_score['compound'] < 0 else 'Neutral'
    print(f"Emotion: {sentiment} (Points: {sentiment_score['compound']})")

    # Customize the system message based on sentiment
    if sentiment == 'Positive':
        prompt = "Be Helpful, Be positive, Be energetic, Be involved, Be concise, Make the user happy! Don't let the user down! Be like a human! Limit to 50 words"
    elif sentiment == 'Negative':
        prompt = "Be Helpful, Be sympathetic, Be understanding, Be concise, The user is having a negative mood and don't hurt his heart again, Be like a human! Limit to 50 words"
    else:
        prompt = "Be Helpful, Be professional, Be active,Be concise,  Focus on the problems and solutions, Be concise and to the point, Limit to 50 words"

    # Create a conversation with the GPT-3.5 Turbo model
    chat_completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=150
    )
    response = chat_completion.choices[0].message.content
    print(response)
