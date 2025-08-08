import pandas as pd
import numpy as np

# --- Load the datasets ---
# These files have some encoding issues, so we use 'latin-1'
# Also, there are some columns with mixed data types, so we'll handle that.
try:
    books = pd.read_csv('data/BX-Books.csv', sep=';', on_bad_lines='skip', encoding='latin-1', low_memory=False)
    users = pd.read_csv('data/BX-Users.csv', sep=';', on_bad_lines='skip', encoding='latin-1')
    ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=';', on_bad_lines='skip', encoding='latin-1')
    print("Files loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the CSV files are in the 'data' directory.")
    exit()

# --- Initial Exploration ---
print("\n--- Books DataFrame ---")
print(books.head()) # Print the first 5 rows
print(books.info()) # Print a concise summary of the DataFrame

print("\n--- Users DataFrame ---")
print(users.head())
print(users.info())

print("\n--- Ratings DataFrame ---")
print(ratings.head())
print(ratings.info())


# --- Data Cleaning & Preprocessing ---

# 1. Clean the 'books' DataFrame
print("\n--- Cleaning Books DataFrame ---")

# Correct column names for easier access
books.columns = ['ISBN', 'Book_Title', 'Book_Author', 'Year_Of_Publication', 'Publisher', 'Image_URL_S', 'Image_URL_M', 'Image_URL_L']

# Convert 'Year_Of_Publication' to numeric, forcing errors to NaN (Not a Number)
books['Year_Of_Publication'] = pd.to_numeric(books['Year_Of_Publication'], errors='coerce')

# Drop rows where 'Year_Of_Publication' is now NaN and filter for sensible years
books.dropna(subset=['Year_Of_Publication'], inplace=True)
books['Year_Of_Publication'] = books['Year_Of_Publication'].astype(int)

# Filter out invalid years (e.g., 0 or future years)
current_year = pd.Timestamp.now().year
books = books[(books['Year_Of_Publication'] > 1900) & (books['Year_Of_Publication'] <= current_year)]

# Fill missing values for author and publisher
books['Book_Author'].fillna('Unknown', inplace=True)
books['Publisher'].fillna('Unknown', inplace=True)

print("Books DataFrame cleaned.")
print(f"Shape of books after cleaning: {books.shape}")


# 2. Clean the 'ratings' DataFrame
print("\n--- Cleaning Ratings DataFrame ---")

# Correct column names
ratings.columns = ['User_ID', 'ISBN', 'Book_Rating']

# Focus on explicit ratings only (ratings from 1 to 10)
# Ratings of 0 are considered "implicit" and we will filter them out for this model.
ratings_explicit = ratings[ratings['Book_Rating'] != 0]

print(f"Shape of ratings before filtering: {ratings_explicit.shape}")

# 3. Filter data for model quality
# To ensure statistical significance, we'll only keep users who have rated at least 5 books
# and books that have been rated by at least 10 users.

# Get counts of ratings per user
user_rating_counts = ratings_explicit['User_ID'].value_counts()
# Get counts of ratings per book
book_rating_counts = ratings_explicit['ISBN'].value_counts()

# Filter users and books
active_users = user_rating_counts[user_rating_counts >= 5].index
popular_books = book_rating_counts[book_rating_counts >= 10].index

ratings_filtered = ratings_explicit[ratings_explicit['User_ID'].isin(active_users)]
ratings_filtered = ratings_filtered[ratings_filtered['ISBN'].isin(popular_books)]

print(f"Shape of ratings after filtering: {ratings_filtered.shape}")


# 4. Merge DataFrames to create the final dataset
print("\n--- Merging DataFrames ---")

# Merge the filtered ratings with book information
final_df = pd.merge(ratings_filtered, books, on='ISBN')

# Drop unnecessary columns
final_df.drop(['Image_URL_S', 'Image_URL_M', 'Image_URL_L'], axis=1, inplace=True)

print("Final DataFrame created.")
print(final_df.head())
print(f"Shape of final DataFrame: {final_df.shape}")


# 5. Save the cleaned data to a new file
# This is a crucial step. We'll use this cleaned file for all subsequent steps.
output_file = 'data/preprocessed_data.csv'
final_df.to_csv(output_file, index=False)

print(f"\nPreprocessing complete. Cleaned data saved to '{output_file}'")