
import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model (lr) and vectorizer (tfidf)
with open('lr_model.pkl', 'rb') as model_file:
    lr = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Streamlit page title and description
st.title('Review Rating Prediction')
st.write("Please enter your product or service review below:")

# User input
user_review = st.text_area("Enter your review:", "", height=150)

# Predict function
def predict_review(review):
    # Preprocess and vectorize the review
    review_vectorized = tfidf.transform([review])  # Vectorizing the review using the pre-loaded vectorizer
    
    # Predict using the preloaded model (lr)
    predicted_rating = lr.predict(review_vectorized)  # Prediction (star rating)
    
    return predicted_rating[0]  # Return the predicted rating (single value)

# Button to trigger prediction
if st.button('Predict'):
    if user_review:
        predicted_rating = predict_review(user_review)
        st.subheader(f"Predicted Rating: {predicted_rating}")
    else:
        st.warning("Please enter a review to get a prediction.")
