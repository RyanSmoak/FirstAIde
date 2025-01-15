import eel
import voice  
from decouple import config

# Initialize Eel
eel.init('Gamecon/web')

@eel.expose
def greet_user():
    greeting = voice.greet_user()
    return greeting

@eel.expose
def take_command():
    response = voice.take_command().lower()
    return response 

   
if __name__ == '__main__':
    eel.start('index.html', size=(800, 600))
