const responseDiv = document.getElementById('response');

async function greetUser() {
    const greeting = await eel.greet_user()();
    responseDiv.innerText = greeting;
}

async function startListening() {
    while (true) {
        const command = await eel.take_command()();
        responseDiv.innerText = command;
        if (command.toLowerCase().includes('exit')) {
            break;
        }
    }
}

// Automatically greet the user and start listening when the page loads
window.onload = async () => {
    greetUser();
    startListening();
}
