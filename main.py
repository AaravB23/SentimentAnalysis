from sentiment_model import load_model

model = load_model()

while True:
    user_input = input("Enter a review (or 'quit'): ")
    if user_input.lower() == 'quit':
        break
    prediction = model.predict([user_input])[0]
    print(f"Predicted sentiment: {prediction.capitalize()}\n")