import pickle
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
from sqlalchemy import create_engine, text

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Database Connection ---
# IMPORTANT: Replace 'your_chosen_password' with the password for 'rec_user'.
db_user = 'rec_user'
db_password = '2369' # <-- CHANGE THIS
db_host = 'localhost'
db_port = '5432'
db_name = 'recommendations_db'

db_connection_str = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(db_connection_str)
print("Database engine created successfully.")

# --- Load the Model Components ---
print("Loading model components...")
try:
    with open('models/sklearn_svd_model.pkl', 'rb') as file:
        model_components = pickle.load(file)
    print("Model components loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'models/sklearn_svd_model.pkl' not found.")
    print("Please run the train_sklearn_model.py script first.")
    exit()

# Unpack the components
svd_model = model_components['svd_model']
user_item_matrix = model_components['user_item_matrix']

# --- Pre-calculate the full predicted ratings matrix ---
print("Calculating the full predicted ratings matrix... This may take a moment.")
user_features_matrix = svd_model.transform(user_item_matrix)
item_features_matrix = svd_model.components_
predicted_ratings = np.dot(user_features_matrix, item_features_matrix)
predicted_ratings_df = pd.DataFrame(
    predicted_ratings,
    index=user_item_matrix.index,
    columns=user_item_matrix.columns
)
print("Predicted ratings matrix ready.")

# --- Define the Routes ---

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    """Takes a user_id and returns a list of book recommendations."""
    if user_id not in predicted_ratings_df.index:
        return jsonify({"error": f"User ID {user_id} not found in the model."}), 404

    num_recommendations = int(request.args.get('count', 10))
    user_predicted_ratings = predicted_ratings_df.loc[user_id].sort_values(ascending=False)

    print(f"Querying database for books rated by user {user_id}...")
    try:
        with engine.connect() as connection:
            query = text("""
                SELECT b."Book_Title" FROM ratings r JOIN books b ON r."ISBN" = b."ISBN"
                WHERE r."User_ID" = :user_id
            """)
            result = connection.execute(query, {"user_id": user_id})
            books_already_rated = [row[0] for row in result]
    except Exception as e:
        print(f"Database query failed: {e}")
        return jsonify({"error": "Could not retrieve user rating history from database."}), 500

    recommendations = user_predicted_ratings.drop(books_already_rated, errors='ignore').head(num_recommendations)
    
    response = {
        "user_id": user_id,
        "recommendations": [{"book_title": title, "predicted_rating": round(rating, 4)} for title, rating in recommendations.items()]
    }
    return jsonify(response)

@app.route('/get_all_books')
def get_all_books():
    """A helper endpoint to provide book titles and ISBNs to the frontend."""
    try:
        with engine.connect() as connection:
            query = text('SELECT "ISBN", "Book_Title" FROM books;')
            result = connection.execute(query)
            all_books = [{"ISBN": row[0], "Book_Title": row[1]} for row in result]
            return jsonify(all_books)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/rate', methods=['POST'])
def rate_book():
    """
    Receives a new rating from the user and stores it in the database.
    This version uses a simple 'delete-then-insert' to avoid ON CONFLICT issues.
    """
    data = request.get_json()
    if not data or 'user_id' not in data or 'isbn' not in data or 'rating' not in data:
        return jsonify({"error": "Missing data. Required: user_id, isbn, rating"}), 400

    user_id = data['user_id']
    isbn = data['isbn']
    rating = data['rating']

    print(f"Received rating for book {isbn} from user {user_id}: {rating}")

    try:
        with engine.connect() as connection:
            with connection.begin(): # Start a transaction
                # First, delete any existing rating for this user and book
                delete_query = text("""
                    DELETE FROM ratings WHERE "User_ID" = :user_id AND "ISBN" = :isbn
                """)
                connection.execute(delete_query, {"user_id": user_id, "isbn": isbn})

                # Then, insert the new rating
                insert_query = text("""
                    INSERT INTO ratings ("User_ID", "ISBN", "Book_Rating")
                    VALUES (:user_id, :isbn, :rating)
                """)
                connection.execute(insert_query, {"user_id": user_id, "isbn": isbn, "rating": rating})
        
        return jsonify({"success": True, "message": "Rating saved successfully."})

    except Exception as e:
        print(f"Database error on rating submission: {e}")
        return jsonify({"error": "Could not save rating to the database."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)