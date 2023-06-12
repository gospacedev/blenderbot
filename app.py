from flask import Flask, render_template, request
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load the Blenderbot-400M-distill model
mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)

app = Flask(__name__)

# Create an empty tuple to store the user_input data
history = ("")
ui_history = []

# Create a function to generate a response to a user"s input
def generate_response(history):
    # Encode the user"s input
    inputs = tokenizer(history, return_tensors="pt")

    # Generate a response
    reply_ids = model.generate(**inputs, max_length=60)

    # Decode the response
    return tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" or "GET":
        global history
        user_input = request.form["user_input"]
        history += tokenizer.bos_token + user_input + tokenizer.eos_token + " "

        response = request.form["response"]
        response = generate_response(history)
        history += tokenizer.bos_token + response + tokenizer.eos_token + " "

        ui_history.append(user_input)
        ui_history.append(response)

    return render_template("index.html", ui_history=ui_history)

if __name__ == "__main__":
    app.run(debug=True)