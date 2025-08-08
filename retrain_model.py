import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.decomposition import TruncatedSVD
import pickle
import os
print("Starting model retraining process...")

# --- 1. Database Connection ---
# IMPORTANT: Replace 'your_chosen_password' with the password you created for 'rec_user'.
db_user = 'rec_user'
db_password = '2369' # <-- CHANGE THIS
db_host = 'localhost'
db_port = '5432'
db_name = 'recommendations_db'

db_connection_str = 'postgresql://neondb_owner:npg_Xuog96jEHaKZ@ep-square-lake-adwmn1o8-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
engine = create_engine(db_connection_str)

# --- 2. Load ALL Data from the Database ---
try:
    print("Loading data from the database...")
    # Read the ratings table into a pandas DataFrame
    ratings_df = pd.read_sql("SELECT * FROM ratings", engine)
    print(f"Loaded {len(ratings_df)} ratings from the database.")

    # Read the books table to get title information
    books_df = pd.read_sql("SELECT * FROM books", engine)
    print(f"Loaded {len(books_df)} books from the database.")

except Exception as e:
    print(f"Failed to load data from database: {e}")
    exit()

# --- 3. Prepare the Final DataFrame ---
# Merge the ratings with the book titles
final_df = pd.merge(ratings_df, books_df, on='ISBN')
print("Data merged successfully.")


# --- 4. Create the User-Item Matrix ---
# This part is identical to the original training script
print("\nCreating the user-item utility matrix...")
user_item_matrix = final_df.pivot_table(
    index='User_ID',
    columns='Book_Title',
    values='Book_Rating'
).fillna(0)

print("User-item matrix created successfully.")
print(f"Shape of the new matrix: {user_item_matrix.shape}")


# --- 5. Decompose the Matrix with TruncatedSVD ---
print("\nRetraining the SVD model on the new data...")
svd = TruncatedSVD(n_components=50, random_state=42)
svd.fit(user_item_matrix)
print("Model retraining completed.")


# --- 6. Save the NEW Model and Data ---
# This will OVERWRITE the old model file with the new, smarter one.
model_components = {
    'svd_model': svd,
    'user_item_matrix': user_item_matrix,
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

print("\nRetraining process complete! The model has been updated.")
print("To see the changes, you must now restart the Flask application (app.py).")
