# Handles reading and managing CSV data (EXPLAIN)
import pandas as pd

# Splits dataset into a training and testing split,
# used for teaching the model and testing it's accuracy. (EXPLAIN)
from sklearn.model_selection import train_test_split

# Converts review into numerical matrix of word counts.
from sklearn.feature_extraction.text import CountVectorizer

# Naive Bayes algorithm for classifying. (EXPLAIN)
from sklearn.naive_bayes import MultinomialNB

# combines preprocessing steps (vectorization) and the model into one reusable object.
from sklearn.pipeline import make_pipeline

# Saves and loads trained models so you don't have to retrain everytime.
import joblib

def train_model():
    # Loads csv as a dataframe (EXPLAIN)
    data = pd.read_csv("data/IMDB_Dataset.csv")

    # X : Text of reviews | Y : Sentiment labels (pos, neg)
    # test_size : 0.2, 20% of data left out for evaluation
    # random_state : 42, fixes randomness to make result reproduceable
    x_train, x_test, y_train, y_test = train_test_split(
        data["review"], data["sentiment"], test_size=0.2, random_state=42
    )

    # CountVectorizer tokenizes (EXPLAIN) text into words by
    # removing punctuation and makes it them all lowercase.
    # Creates sparse matrix (EXPLAIN), each column associating
    # to a word and the row holding the count of that word.
    # 
    # Multinomial NB (EXPLAIN) takes the word counts and learns how 
    # likely each word is to appear in positive vs negative reviews.
    #
    # Pipeline lets you create both objects as one, ex when you
    # call fit() or predict() both work in the correct order.
    # 
    # This is staging the model to train on our data.
    model = make_pipeline(CountVectorizer(), MultinomialNB())

    # Trains the pipeline. The CountVectorizer learns from x_train,
    # and MultinomialNB learns probabilities for each word given
    # label word and pos or neg.
    # The model can now take any new text and output a prediction.
    print("Training model...")
    model.fit(x_train, y_train)
    print("Finished training.")
    # Shows performance on unseen data.
    # .score() calls predict(X_test) and compares
    # to y_test, returning the fraction of correct 
    # predictions. Good feedback for model accuracy.
    print("Accuracy: ", model.score(x_test, y_test))


    # Serializes the trained pipeline to
    # binary .pkl file (EXPLAIN). Allows for
    # reuse without retraining. 
    joblib.dump(model, "sentiment_model.pkl")

def load_model():
    # Helper function that loads the trained pipeline.
    # This contains the vectorizer and classifier (EXPLAIN)
    # so all you need to run is model.predict(["I loved it!"])
    return joblib.load("sentiment_model.pkl")

# If ran directly, retrain model, otherwise ignore.
if __name__ == "__main__":
    train_model()