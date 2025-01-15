from flask import Flask, request, jsonify, render_template
import torch
from main import BigramLanguageModel, encode, decode  # Import your model and utilities

# Initialize the app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BigramLanguageModel().to(device)
model.load_state_dict(torch.load('aide_model.pth', map_location=device))
model.eval()

@app.route("/")
def index():
    """Serve the index.html page."""
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    """Handle text generation requests."""
    data = request.get_json()  # Get the JSON data from the POST request
    starting_text = data.get("text", "")  # Extract the 'text' field from the JSON
    
    if not starting_text:
        return jsonify({"error": "No input text provided"}), 400  # If no text is provided, return error
    
    try:
        # Encode the starting text and generate the output
        starting_context = torch.tensor([encode(starting_text)], dtype=torch.long, device=device)
        print(f"Starting context tensor: {starting_context}")
        generated_output = decode(model.generate(starting_context, max_new_tokens=305)[0].tolist())
        print(f"Generated output: {generated_output}")
        return jsonify({"output": generated_output})  # Return the generated output as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle any errors during generation

if __name__ == "__main__":
    app.run(debug=True)
