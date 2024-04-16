import os
from openai import OpenAI
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import gradio as gr
from google.cloud import storage

# Vader Dictionary
nltk.download('vader_lexicon')

# Initialize Vader
sia = SentimentIntensityAnalyzer()

client = OpenAI(
    api_key= os.getenv('API_KEY'),
)

messages = []

def write_transcript_to_gcs(text):
    storage_client = storage.Client()
    bucket_name = "transcripts_534_bucket"

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('transcripts.txt')

    if blob.exists():
        current_contents = blob.download_as_text()
        new_contents = current_contents + text
    else:
        new_contents = text

    blob.upload_from_string(new_contents, content_type='text/plain')

def update_system_message(sentiment):

    # Customize the system message based on sentiment
    if sentiment == 'Positive':
        prompt = "You are an IT assistant. Be Helpful, Be positive, Be energetic, Be involved, Be concise, as the user is already in a good mood! Don't let the user down! Be like a human! Limit to 50 words"
    elif sentiment == 'Negative':
        prompt = "You are an IT assistant. Be Helpful, Be sympathetic, Be understanding, Be concise, The user is having a negative mood, so be understanding and calming. Be like a human! Limit to 50 words"
    else:
        prompt = "You are an IT assistant. Be Helpful, Be professional, Be active,Be concise, Focus on the problems and solutions, as your client is in a neutral mood. Be concise and to the point, Limit to 50 words"
    return {"role": "system", "content": prompt}

def create_initial_message(initial_input):
    # Perform sentiment analysis
    sentiment_score = sia.polarity_scores(initial_input)
    sentiment = 'Positive' if sentiment_score['compound'] > 0 else 'Negative' if sentiment_score['compound'] < 0 else 'Neutral'
    print(f"Emotion: {sentiment} (Points: {sentiment_score['compound']})")

    system_message = update_system_message(sentiment)
    return [
        system_message,
        {"role": "user", "content": initial_input}
    ]

def continue_conversation():
    global messages
    # Access the most recent user message
    most_recent_user_message = next((msg for msg in reversed(messages) if msg['role'] == 'user'), None)
    sentiment = "Neutral"
    if most_recent_user_message:
        user_message_content = most_recent_user_message['content']
        sentiment_score = sia.polarity_scores(user_message_content)
        sentiment = 'Positive' if sentiment_score['compound'] > 0 else 'Negative' if sentiment_score['compound'] < 0 else 'Neutral'
        print(f"Emotion: {sentiment} (Points: {sentiment_score['compound']})")
        write_transcript_to_gcs(f"Sentiment {sentiment}, Score: {sentiment_score}\n")




    messages[0] = update_system_message(sentiment)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response

def chatbot(input):
    # Start the conversation with the initial message
    global messages
    if len(messages) == 0:
        messages = create_initial_message(input)
        write_transcript_to_gcs(" \n\n\n NEW USER \n\n")
    
    messages.append({"role": "user", "content": input})
    response = continue_conversation()
    assistant_message = response.choices[0].message.content
    print("Assistant:", assistant_message)

    # Append the user and assistant messages to the conversation history
    messages.append({"role": "assistant", "content": assistant_message})

    write_transcript_to_gcs(f"User: {input}\nBot: {assistant_message}\n\n")
    
    return assistant_message

interface = gr.Interface(fn=chatbot,
                         inputs=gr.Textbox(lines=2, placeholder="Type something..."),
                         outputs=gr.Textbox())

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 7860))  # Default to 7860 if PORT not set
    interface.launch(server_name='0.0.0.0', server_port=port)
