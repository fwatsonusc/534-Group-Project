import os
from openai import OpenAI
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import gradio as gr
from google.cloud import storage
import random

# Vader Dictionary
nltk.download('vader_lexicon')

# Initialize Vader
sia = SentimentIntensityAnalyzer()

client = OpenAI(
    api_key= os.getenv('API_KEY'),
)

messages = []
chat_history = []
participant_id = "Your participant ID will show up here."

def write_transcript_to_gcs(text):
    storage_client = storage.Client()
    bucket_name = "transcripts_534_bucket"

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('transcripts-unemotional.txt')

    if blob.exists():
        current_contents = blob.download_as_text()
        new_contents = current_contents + text
    else:
        new_contents = text

    blob.upload_from_string(new_contents, content_type='text/plain')

def update_system_message(sentiment):

   
    prompt = "You are an IT assistant. Be helpful, professional, and active. Focus on the problems and solutions without making adjustments for how you perceive the client's emotion, always assuming that your client is in a neutral mood. Be concise and to the point, Limit to 50 words"
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

def chatbot(input, state, start_over):
    # Start the conversation with the initial message
    global messages
    global chat_history
    global participant_id
    # if start_over:
    #     chat_history = []  # Reset the state
    #     write_transcript_to_gcs(" \n\n\n NEW USER \n\n")
    #     start_over = False
    # if chat_history is None:
    #     chat_history = []  # Initialize state if it's the first call

    if state[0] == "initialize":
        chat_history = []  # Reset the state
        participant_id = str(random.randint(1000000000, 9999999999))
        state[0] = ""
        state[1] = participant_id
        write_transcript_to_gcs(f" \n\n\n NEW USER - {participant_id} \n\n")
        messages = create_initial_message(input)
    
    messages.append({"role": "user", "content": input})
    response = continue_conversation()
    assistant_message = response.choices[0].message.content
    print("Assistant:", assistant_message)

    # Append the user and assistant messages to the conversation history
    messages.append({"role": "assistant", "content": assistant_message})
    chat_history.append(f"User: {input}")
    chat_history.append(f"Bot: {assistant_message}")

    write_transcript_to_gcs(f"User: {input}\nBot: {assistant_message}\n\n")
    formatted_output = "\n".join(chat_history)
    
    return chat_history, formatted_output, participant_id, False

interface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Textbox(label="Your Message"),
        gr.State(value=["initialize", ""]),  # State input to maintain history
        gr.Checkbox(label="Start Over", value=True, visible=False),
    ],
    outputs=[
        gr.State(),  # State output to update history
        gr.Textbox(label="Conversation", lines=10),  # Show the conversation history
        gr.Label(label="ParticipantID"),
        gr.Checkbox(visible=False),

    ],
    title="IT Assistant Chatbot",
    allow_flagging=False,
    live=False
)

# interface = gr.Interface(fn=chatbot,
#                          inputs=gr.Textbox(lines=2, placeholder="Type something..."),
#                          outputs=gr.Textbox())

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 7860))  # Default to 8080 if PORT not set
    interface.launch(server_name='0.0.0.0', server_port=port)
