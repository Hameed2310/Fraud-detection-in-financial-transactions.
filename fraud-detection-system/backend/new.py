import pandas as pd
import numpy as np
import librosa
import sounddevice as sd
import speech_recognition as sr
import pyttsx3
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# 1. Load Datasets
text_df = pd.read_csv("/mnt/data/text.csv")
speech_df = pd.read_csv("/mnt/data/speech_emotions.csv")

# 2. Text Emotion Classifier
tfidf = TfidfVectorizer()
X_text = tfidf.fit_transform(text_df['text'])
y_text = text_df['emotion']

X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.2)
text_model = RandomForestClassifier()
text_model.fit(X_train_text, y_train_text)

# 3. Audio Emotion Classifier (using MFCC features)
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

X_audio = np.array([extract_features(row['path']) for _, row in speech_df.iterrows()])
y_audio = speech_df['emotion']

X_train_audio, X_test_audio, y_train_audio, y_test_audio = train_test_split(X_audio, y_audio, test_size=0.2)
audio_model = RandomForestClassifier()
audio_model.fit(X_train_audio, y_train_audio)

# 4. Text-to-Speech Engine
engine = pyttsx3.init()

def speak(text):
    print("Chatbot:", text)
    engine.say(text)
    engine.runAndWait()

# 5. Speech Recognition (Mic Input)
def record_and_predict_emotion(duration=4):
    fs = 44100
    print("Listening...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    librosa.output.write_wav("temp.wav", recording.T[0], sr=fs)
    features = extract_features("temp.wav").reshape(1, -1)
    return audio_model.predict(features)[0]

# 6. Text Prediction
def predict_text_emotion(user_input):
    vect_input = tfidf.transform([user_input])
    return text_model.predict(vect_input)[0]

# 7. Chatbot Logic
emotion_responses = {
    "happy": "That's wonderful to hear!",
    "sad": "I'm here for you. It's okay to feel this way.",
    "angry": "Take a deep breath. Want to talk about it?",
    "neutral": "Got it. Tell me more.",
    "fear": "It must be tough. You're not alone.",
    "disgust": "That sounds unpleasant. I'm sorry you had to go through that.",
    "surprise": "Oh! That’s unexpected. Tell me more!"
}

# 8. Main Chat Loop
def chatbot():
    speak("Hello! Would you like to talk via text or voice?")
    mode = input("Enter 'text' or 'voice': ").strip().lower()

    while True:
        if mode == "text":
            user_input = input("You: ")
            if user_input.lower() == 'bye':
                speak("Goodbye!")
                break
            emotion = predict_text_emotion(user_input)
        elif mode == "voice":
            emotion = record_and_predict_emotion()
        else:
            speak("Invalid mode. Try again.")
            break

        response = emotion_responses.get(emotion, "I'm not sure how to respond to that.")
        speak(response)

# Run chatbot
chatbot()
