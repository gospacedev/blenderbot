from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load the Blenderbot-400M-distill model
mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)

# Store user input and bot responses
history = ("")

# Create a function to generate a response to a user's input
def generate_response(user_input):
    # Encode the user's input
    inputs = tokenizer(user_input, return_tensors="pt")

    # Generate a response
    reply_ids= model.generate(**inputs, max_length=60)

    # Decode the response
    return tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

# Start a conversation with the user
while True:
    user_input = input("User: ")
    history += tokenizer.bos_token + user_input + tokenizer.eos_token + " "
    
    response = generate_response(user_input)
    
    history += tokenizer.bos_token + response + tokenizer.eos_token + " "
    print("Bot: " + response)

    print(history)