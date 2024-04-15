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
    # Start with the first system message
    # Perform sentiment analysis
    sentiment_score = sia.polarity_scores(initial_input)
    sentiment = 'Positive' if sentiment_score['compound'] > 0 else 'Negative' if sentiment_score['compound'] < 0 else 'Neutral'
    print(f"Emotion: {sentiment} (Points: {sentiment_score['compound']})")

    system_message = update_system_message(sentiment)
    return [
        system_message,
        {"role": "user", "content": initial_input}
    ]

def continue_conversation(messages):

    # Access the most recent user message
    most_recent_user_message = next((msg for msg in reversed(messages) if msg['role'] == 'user'), None)
    sentiment = "Neutral"
    if most_recent_user_message:
        user_message_content = most_recent_user_message['content']
        
        # Update the system message before sending it to the model
        sentiment_score = sia.polarity_scores(user_message_content)
        sentiment = 'Positive' if sentiment_score['compound'] > 0 else 'Negative' if sentiment_score['compound'] < 0 else 'Neutral'
        print(f"Emotion: {sentiment} (Points: {sentiment_score['compound']})")




    messages[0] = update_system_message(sentiment)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response

def main():
    # Start the conversation with the initial message
    intial_input = input("Input: ")
    messages = create_initial_message(intial_input)
    
    # Enter an infinite loop to handle conversation
    while True:
        # Generate the assistant's response
        response = continue_conversation(messages)
        assistant_message = response.choices[0].message.content
        print("Assistant:", assistant_message)
        
        # Ask for user input
        user_input = input("Input: ")
        if user_input.lower() == 'exit':
            print("Exiting the conversation.")
            break
        
        # Append the user and assistant messages to the conversation history
        messages.append({"role": "assistant", "content": assistant_message})
        messages.append({"role": "user", "content": user_input})

if __name__ == "__main__":
    main()

# while True:
#     user_input = input("Input: ")
#     if user_input.lower() == "Exit":
#         print("The chat is over。")
#         break

#     # Perform sentiment analysis
#     sentiment_score = sia.polarity_scores(user_input)
#     sentiment = 'Positive' if sentiment_score['compound'] > 0 else 'Negative' if sentiment_score['compound'] < 0 else 'Neutral'
#     print(f"Emotion: {sentiment} (Points: {sentiment_score['compound']})")


#     # Create a conversation with the GPT-3.5 Turbo model
#     chat_completion = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": prompt},
#             {"role": "user", "content": user_input}
#         ],
#         temperature=0.7,
#         max_tokens=150
#     )
#     response = chat_completion.choices[0].message.content
#     print(response)
