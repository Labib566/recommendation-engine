import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import pickle
import os # Import the os module

print("Loading preprocessed data...")
try:
    df = pd.read_csv('data/preprocessed_data.csv')
except FileNotFoundError:
    print("Error: 'preprocessed_data.csv' not found.")
    print("Please run the data_preprocessing.py script first.")
    exit()

print("Data loaded successfully.")

# --- 1. Create the User-Item Matrix ---
print("\nCreating the user-item utility matrix...")

# We use pivot_table to transform the data.
# Index will be the users, columns will be the book titles, and values will be the ratings.
# Using 'Book_Title' is more intuitive than 'ISBN' for columns.
user_item_matrix = df.pivot_table(index='User_ID', columns='Book_Title', values='Book_Rating').fillna(0)

print("User-item matrix created successfully.")
print(f"Shape of the matrix: {user_item_matrix.shape}")


# --- 2. Decompose the Matrix with TruncatedSVD ---
print("\nDecomposing the matrix with TruncatedSVD...")

# Create the SVD model. n_components is the number of latent features to find.
# 50 is a common starting point.
svd = TruncatedSVD(n_components=50, random_state=42)

# Fit the model to our data
matrix_decomposed = svd.fit_transform(user_item_matrix)

print("Matrix decomposed successfully.")


# --- 3. Predict Ratings and Generate Recommendations ---
# To get predicted ratings, we multiply the decomposed matrix by the SVD components.
# This reconstructs the matrix, filling in the zeros with predicted ratings.
predicted_ratings = np.dot(matrix_decomposed, svd.components_)

# Create a DataFrame with the predicted ratings
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

print("\n--- Recommendation Example ---")

def recommend_books(user_id, num_recommendations=5):
    """Recommends books for a given user."""
    # Get the user's predicted ratings
    user_predicted_ratings = predicted_ratings_df.loc[user_id].sort_values(ascending=False)

    # Get the books the user has already rated
    user_rated_books = user_item_matrix.loc[user_id]
    user_rated_books = user_rated_books[user_rated_books > 0].index

    # Find recommendations that are not in the books the user has already rated
    recommendations = user_predicted_ratings.drop(user_rated_books).head(num_recommendations)

    return recommendations

# Test with a sample user (e.g., the first user in our matrix)
sample_user_id = user_item_matrix.index[0]
recommendations = recommend_books(sample_user_id, 10)

print(f"Top 10 book recommendations for User ID {sample_user_id}:")
print(recommendations)


# --- 4. Save the Model and Data ---
# We need to save the SVD model itself, and also the columns (book titles)
# and index (user IDs) of our utility matrix to interpret the results later.
model_components = {
    'svd_model': svd,
    'user_item_matrix': user_item_matrix,
    'predicted_ratings_df': predicted_ratings_df,
    'user_item_matrix_columns': user_item_matrix.columns,
    'user_item_matrix_index': user_item_matrix.index
}

# Define the directory where you want to save the model
model_dir = 'models' # You can change this to any folder name you prefer

# Create the directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Construct the full path for the model file
model_filename = os.path.join(model_dir, 'sklearn_svd_model.pkl')
print(f"\nSaving model components to {model_filename}...")

with open(model_filename, 'wb') as file:
    pickle.dump(model_components, file)

print("Model components saved successfully.")
