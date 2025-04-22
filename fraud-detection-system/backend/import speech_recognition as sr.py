# Simple chatbot for beginners that responds to your emotion

# Predefined emotional word lists
positive_words = ["happy", "great", "awesome", "good", "love", "fantastic"]
negative_words = ["sad", "bad", "terrible", "angry", "hate", "upset"]

# Start the chatbot
print("🤖 Hi, I'm your chatbot friend!")
print("Type 'bye' to exit.\n")

while True:
    user_input = input("You: ").lower()

    if "bye" in user_input:
        print("🤖 Goodbye! Take care 😊")
        break

    # Check for emotion
    if any(word in user_input for word in positive_words):
        print("🤖 I'm glad to hear that! 😊")
    elif any(word in user_input for word in negative_words):
        print("🤖 I'm sorry you're feeling that way. I'm here for you. ❤️")
    else:
        print("🤖 Thanks for sharing! Tell me more.")