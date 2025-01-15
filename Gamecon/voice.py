import speech_recognition as sr
import os
import subprocess as sp
from datetime import datetime
from decouple import config
from random import choice
from conversation import random_text
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

USER = config('USER')
HOSTNAME = config('BOT')
FIRST = config('FIRST')
SECOND = config('SECOND')
THIRD = config('THIRD')
ALL = config('ALL')


def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = 'voice.mp3'
    tts.save(filename)
    sound = AudioSegment.from_file(filename)
    play(sound)
    os.remove(filename)

def greet_user():
    hour = datetime.now().hour
    if 6 <= hour < 12:
        greet = f"Good morning {USER}"
    elif 12 <= hour <= 16:
        greet = f"Good afternoon {USER}"
    elif 16 <= hour < 19:
        greet = f"Good evening {USER}"
    else:
        greet = f"Hello {USER}"
    greet += f", I am {HOSTNAME}. How may I assist you?"
    speak(greet)
    return greet

def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}")

        if 'stop' in query or 'exit' in query:
            hour = datetime.now().hour
            if 21 <= hour < 6:
                farewell = "Good night, take care!"
            else:
                farewell = "Have a good day!"
            speak(farewell)
            return farewell
            exit()
        
        response = process_command(query)
        return response

    except sr.UnknownValueError:
        speak("Sorry, I couldn't understand. Can you please repeat that?")
        return "Sorry, I couldn't understand. Can you please repeat that?"

    except sr.RequestError:
        speak("Sorry, I'm having trouble connecting to the recognition service.")
        return "Sorry, I'm having trouble connecting to the recognition service."

def process_command(query):
    response = ""
    query = query.lower()
    
    if "what can you do" in query:
        response = "I will give you first aid information on burns. Just state the degree of burn whether it is First ,second or third."
        speak(response)
    
    elif "first" in query:
        response = f"{FIRST}"
        speak(response)
    
    elif "second" in query:
        response = f"{SECOND}"
        speak(response)

    elif "third" in query:
        response = f"{THIRD}"
        speak(response)   

    elif "all" in query:
        response = f"{ALL}"
        speak(response) 

    elif "who created you" in query:
        response = "Elle Mibei from the University of Nairobi built me for her second year project ,isn't she cool?"
        speak(response)
    

    return response

# Example usage:
if __name__ == "__main__":
    greet_user()
    while True:
        command = take_command()
        if 'stop' in command or 'exit' in command:
            break
