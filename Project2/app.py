
import streamlit as st
import pickle
import numpy as np
import joblib

# Load the pre-trained models and encoders
with open('rating_models/ord_model_tfidf_downsampled_df.pkl', 'rb') as model_file:
    ord = pickle.load(model_file)

with open('tfidf_downsampled_vectorizer.pkl', 'rb') as vectorizer_file:
    downsampled_tfidf = pickle.load(vectorizer_file)

rf = joblib.load('product_models/rf_model_tfidf_downsampled_df.pkl')

with open('label_encoder_downsampled.pkl', 'rb') as encoder_file:
    label_encoder_rf = pickle.load(encoder_file)

# Streamlit page title and description
st.title('Review Rating and Product Category Prediction')
st.write("Please enter your product or service review below:")

# User input
user_review = st.text_area("Enter your review:", "", height=150)

# Predict function using Ordinal Regression (for traditional method)
def predict_review_ord(review):
    review_vectorized = downsampled_tfidf.transform([review])  # Vectorizing the review using the pre-loaded vectorizer
    predicted_rating = ord.predict(review_vectorized)  # Prediction (star rating)
    return predicted_rating[0]  # Return the predicted rating (single value)

# Predict function using Random Forest (for product classification)
def predict_review_rf(review):
    review_vectorized = downsampled_tfidf.transform([review])  # Vectorize the review
    
    # Validate the loaded model
    if not hasattr(rf, 'predict_proba'):
        raise AttributeError("The loaded Random Forest model is invalid. Ensure it's correctly saved and loaded.")
    
    # Get predicted probabilities for each class
    predicted_probabilities = rf.predict_proba(review_vectorized)[0]
    
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class = label_encoder_rf.inverse_transform([predicted_class_index])[0]
    
    # Get the confidence level (probability of the predicted class)
    confidence = predicted_probabilities[predicted_class_index] * 100  # Convert to percentage
    return predicted_class, confidence

# Button to trigger prediction
if st.button('Predict'):
    if user_review:
        # Predict rating using Ordinal Regression
        predicted_rating = predict_review_ord(user_review)
        
        # Predict product category using Random Forest and get confidence
        predicted_class, confidence = predict_review_rf(user_review)
        
        # Display both predictions and confidence
        st.subheader(f"Predicted Rating (Ordinal Regression): {predicted_rating}")
        st.subheader(f"Predicted Product Category (Random Forest): {predicted_class}")
        st.write(f"Product confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter a review to get a prediction.")

