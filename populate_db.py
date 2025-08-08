import pandas as pd
from sqlalchemy import create_engine, text

print("Starting database population script...")

# --- Database Connection Details ---
# IMPORTANT: Replace 'your_chosen_password' with the password you created for the 'rec_user'.
db_user = 'rec_user'
db_password = '2369' # <-- CHANGE THIS
db_host = 'localhost'
db_port = '5432'
db_name = 'recommendations_db'

db_connection_str = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(db_connection_str)

# --- Load the Preprocessed Data ---
try:
    df = pd.read_csv('data/preprocessed_data.csv')
    print("Loaded preprocessed_data.csv successfully.")
except FileNotFoundError:
    print("Error: 'preprocessed_data.csv' not found. Please run data_preprocessing.py first.")
    exit()

# --- Create Normalized Tables ---
books = df[['ISBN', 'Book_Title', 'Book_Author', 'Year_Of_Publication', 'Publisher']].copy().drop_duplicates(subset='ISBN')
print(f"Found {len(books)} unique books.")

users = pd.DataFrame(df['User_ID'].unique(), columns=['User_ID'])
print(f"Found {len(users)} unique users.")

ratings = df[['User_ID', 'ISBN', 'Book_Rating']].copy()
print(f"Found {len(ratings)} ratings.")

# --- Write DataFrames to PostgreSQL ---
try:
    print("\nWriting to database... This may take a few minutes.")
    
    books.to_sql('books', engine, if_exists='replace', index=False)
    print("- 'books' table created successfully.")

    users.to_sql('users', engine, if_exists='replace', index=False)
    print("- 'users' table created successfully.")

    ratings.to_sql('ratings', engine, if_exists='replace', index=False)
    print("- 'ratings' table created successfully.")

    # --- Add Primary and Foreign Keys for Data Integrity ---
    print("\nAdding primary and foreign keys...")
    with engine.connect() as connection:
        # In SQLAlchemy 2.0, a transaction is automatically begun.
        connection.execute(text('ALTER TABLE books ADD PRIMARY KEY ("ISBN");'))
        connection.execute(text('ALTER TABLE users ADD PRIMARY KEY ("User_ID");'))
        # Add a composite primary key to ratings to ensure one rating per user-book pair
        connection.execute(text('ALTER TABLE ratings ADD PRIMARY KEY ("User_ID", "ISBN");'))
        connection.execute(text('ALTER TABLE ratings ADD CONSTRAINT fk_user FOREIGN KEY ("User_ID") REFERENCES users("User_ID");'))
        connection.execute(text('ALTER TABLE ratings ADD CONSTRAINT fk_book FOREIGN KEY ("ISBN") REFERENCES books("ISBN");'))
        # The transaction is automatically committed when the 'with' block exits without an error.
    print("- Keys added successfully.")

    print("\nDatabase population complete!")

except Exception as e:
    print(f"\nAn error occurred: {e}")