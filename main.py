from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load the Blenderbot-400M-distill model
mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)

chat_history = ("")

# Create a function to generate a response to a user's input
def generate_response(user_input):
    # Encode the user's input
    inputs = tokenizer(user_input, return_tensors="pt")

    # Generate a response
    reply_ids= model.generate(**inputs)

    # Decode the response
    return tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

# Start a conversation with the user
while True:
    user_input = input("User: ")
    chat_history += user_input
    
    response = generate_response(chat_history)
    
    chat_history += response
    print("Bot: " + response)
