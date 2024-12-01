import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import random

random.seed(42)
np.random.seed(42)
# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load data
print("Loading dataset...")
books = pd.read_csv('Books.csv')
ratings = pd.read_csv('Ratings.csv')

# 1. Retain users with enough ratings (at least 5 ratings)
user_counts = ratings['User-ID'].value_counts()
active_users = user_counts[user_counts >= 5].index
filtered_ratings = ratings[ratings['User-ID'].isin(active_users)]

# 2. Retain books that have been rated by enough users (at least 5 users)
book_counts = filtered_ratings['ISBN'].value_counts()
popular_books = book_counts[book_counts >= 5].index
filtered_ratings = filtered_ratings[filtered_ratings['ISBN'].isin(popular_books)]

# 3. Retain only high-rated data (ratings above 7)
filtered_ratings = filtered_ratings[filtered_ratings['Book-Rating'] >= 7]

# Sample data
sampled_ratings = filtered_ratings.sample(n=5000, random_state=42)
sampled_book_isbns = sampled_ratings['ISBN'].unique()

# Create ISBN to index mapping
isbn_to_idx = {isbn: idx for idx, isbn in enumerate(books['ISBN'])}
idx_to_isbn = {idx: isbn for isbn, idx in isbn_to_idx.items()}

#  Step 2: Prepare data in the format required by the Surprise library
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(sampled_ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

# Split into training and testing datasets
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

# Step 3: Use SVD model
print("Training SVD model...")
algo = SVD()
algo.fit(trainset)

# Step 4: Use BERT model to get book title embeddings
def get_bert_embeddings(titles):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    embeddings = []
    for title in tqdm(titles, desc="Tokenizing titles"):
        if isinstance(title, str):  # Ensure the title is a string
            inputs = tokenizer(title, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        else:
            # For non-string titles, use zero vectors
            embeddings.append(np.zeros((1, 768)))
    
    return np.vstack(embeddings)

# Get BERT embeddings for all book titles
print("Getting BERT embeddings for book titles...")
# Process only the sampled books
sampled_books = books[books['ISBN'].isin(sampled_book_isbns)]
book_titles = sampled_books['Book-Title'].tolist()
book_embeddings = get_bert_embeddings(book_titles)

# Compute cosine similarity matrix for book titles
cosine_sim = cosine_similarity(book_embeddings, book_embeddings)

# Step 5: Define content-based recommendation function
def recommend_books_based_on_content(user_id, num_recommendations=5):
    # Get books rated by the user
    user_rated_books = sampled_ratings[sampled_ratings['User-ID'] == user_id]['ISBN'].tolist()
    
    # Get indices of rated books in the sampled dataset
    rated_books_indices = []
    for isbn in user_rated_books:
        if isbn in sampled_books['ISBN'].values:
            idx = sampled_books[sampled_books['ISBN'] == isbn].index[0]
            rated_books_indices.append(idx - sampled_books.index[0])  # Adjust index to match cosine_sim matrix
    
    # Store all recommendations
    content_based_recommendations = []
    
    for idx in rated_books_indices:
        if idx < len(cosine_sim):  # Ensure the index is within range
            similar_books = list(enumerate(cosine_sim[idx]))
            similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)
            
            for book_idx, sim_score in similar_books[1:]:  # Skip itself
                if book_idx < len(sampled_books):
                    isbn = sampled_books.iloc[book_idx]['ISBN']
                    if isbn not in user_rated_books:
                        content_based_recommendations.append((isbn, sim_score))
    
    # Sort by similarity and return recommendations
    content_based_recommendations = sorted(content_based_recommendations, key=lambda x: x[1], reverse=True)
    recommended_isbns = [isbn for isbn, _ in content_based_recommendations[:num_recommendations]]
    recommended_books = sampled_books[sampled_books['ISBN'].isin(recommended_isbns)]
    
    return recommended_books

# Step 6: Define collaborative filtering recommendation function
def recommend_books_for_user(user_id, num_recommendations=5):
    user_rated_books = sampled_ratings[sampled_ratings['User-ID'] == user_id]['ISBN'].tolist()
    books_to_predict = [isbn for isbn in sampled_book_isbns if isbn not in user_rated_books]
    
    predicted_ratings = []
    for isbn in books_to_predict:
        predicted_ratings.append((isbn, algo.predict(user_id, isbn).est))
    
    top_books = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:num_recommendations]
    recommended_books = sampled_books[sampled_books['ISBN'].isin([isbn for isbn, _ in top_books])]
    return recommended_books

# Step 7: Hybrid recommendation function
def hybrid_recommend_books(user_id, num_recommendations=5, alpha=0.5):
    try:
        if user_id not in sampled_ratings['User-ID'].unique():
            # Get the 5 books with the highest average rating
            top_books = sampled_books.sort_values('Book-Rating', ascending=False).head(num_recommendations)
            return top_books[['ISBN', 'Book-Title', 'Book-Author']]
        collaborative_recommendations = recommend_books_for_user(user_id, num_recommendations)
        content_based_recommendations = recommend_books_based_on_content(user_id, num_recommendations)
        
        # merge recommendation result
        all_recommendations = pd.concat([collaborative_recommendations, content_based_recommendations])
        all_recommendations = all_recommendations.drop_duplicates(subset=['ISBN'])
        try:
            return all_recommendations.sample(num_recommendations,random_state=42)
        except Exception as e:
            return all_recommendations.head(num_recommendations)
    except Exception as e:
        print(f"Error in hybrid_recommend_books for user {user_id}: {str(e)}")
        return pd.DataFrame(columns=['ISBN', 'Book-Title', 'Book-Author'])