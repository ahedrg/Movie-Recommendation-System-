import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox

# Sample user ratings and preferences
user_ratings = {
    'user1': {'The Matrix': 9.0, 'Inception': 8.5, 'Interstellar': 8.0},
    'user2': {'John Wick': 8.5, 'Avengers': 9.0, 'Iron Man': 8.0},
    'user3': {'Inception': 9.0, 'Tenet': 8.5, 'Interstellar': 9.0},
    'user4': {'Doctor Strange': 8.0, 'Avengers': 8.5, 'Captain America': 8.0}
}

movie_data = {
    'title': [
        'The Matrix', 'John Wick', 'Avengers', 'Iron Man', 'Inception',
        'Interstellar', 'Doctor Strange', 'Captain America', 'Tenet'
    ],
    'genre': [
        ['Sci-Fi', 'Action'], ['Action', 'Crime'], ['Action', 'Superhero'],
        ['Superhero', 'Sci-Fi'], ['Sci-Fi', 'Thriller'], ['Sci-Fi', 'Space'],
        ['Fantasy', 'Superhero'], ['Action', 'Superhero'], ['Sci-Fi', 'Thriller']
    ],
    'keywords': [
        ['virtual reality', 'dystopia', 'ai'], ['revenge', 'assassin', 'action'],
        ['team', 'aliens', 'heroes'], ['technology', 'hero', 'genius'],
        ['dreams', 'heist', 'mind'], ['space', 'time travel', 'physics'],
        ['magic', 'multiverse', 'mystic'], ['war', 'super soldier', 'shield'],
        ['time inversion', 'espionage', 'physics']
    ],
    'director': [
        'Wachowskis', 'Chad Stahelski', 'Joss Whedon', 'Jon Favreau',
        'Christopher Nolan', 'Christopher Nolan', 'Scott Derrickson',
        'Joe Johnston', 'Christopher Nolan'
    ],
    'year': [1999, 2014, 2012, 2008, 2010, 2014, 2016, 2011, 2020],
    'popularity': [95, 85, 90, 88, 92, 89, 82, 84, 86]
}

class HybridRecommender:
    def __init__(self, movie_data, user_ratings):
        self.movies_df = pd.DataFrame(movie_data)
        self.user_ratings = user_ratings
        self.similarity_matrix = None
        self.user_profiles = defaultdict(dict)
        self._prepare_data()

    def _prepare_data(self):
        # Create genre and keyword vectors
        self._create_feature_vectors()
        # Calculate movie similarity matrix
        self._calculate_similarity()
        # Build user profiles
        self._build_user_profiles()

    def _create_feature_vectors(self):
        # Create binary vectors for genres and keywords
        all_genres = set(sum(self.movies_df['genre'], []))
        all_keywords = set(sum(self.movies_df['keywords'], []))
        
        for feature, feature_set in [('genre', all_genres), ('keywords', all_keywords)]:
            for item in feature_set:
                self.movies_df[f'{feature}_{item}'] = self.movies_df[feature].apply(
                    lambda x: 1 if item in x else 0
                )

    def _calculate_similarity(self):
        # Combine different features for similarity calculation
        feature_cols = [col for col in self.movies_df.columns 
                       if col.startswith(('genre_', 'keywords_'))]
        
        # Add normalized year and popularity
        scaler = MinMaxScaler()
        self.movies_df['year_norm'] = scaler.fit_transform(
            self.movies_df[['year']]
        )
        self.movies_df['popularity_norm'] = scaler.fit_transform(
            self.movies_df[['popularity']]
        )
        
        feature_cols.extend(['year_norm', 'popularity_norm'])
        features_matrix = self.movies_df[feature_cols].values
        self.similarity_matrix = cosine_similarity(features_matrix)

    def _build_user_profiles(self):
        for user, ratings in self.user_ratings.items():
            genre_prefs = defaultdict(float)
            keyword_prefs = defaultdict(float)
            
            for movie, rating in ratings.items():
                idx = self.movies_df[self.movies_df['title'] == movie].index[0]
                movie_genres = self.movies_df.loc[idx, 'genre']
                movie_keywords = self.movies_df.loc[idx, 'keywords']
                
                for genre in movie_genres:
                    genre_prefs[genre] += rating
                for keyword in movie_keywords:
                    keyword_prefs[keyword] += rating
            
            self.user_profiles[user] = {
                'genres': dict(genre_prefs),
                'keywords': dict(keyword_prefs)
            }

    def get_recommendations(self, user_id=None, movie_title=None, n_recommendations=3):
        if user_id and movie_title:
            return self._get_hybrid_recommendations(user_id, movie_title, n_recommendations)
        elif user_id:
            return self._get_user_based_recommendations(user_id, n_recommendations)
        elif movie_title:
            return self._get_content_based_recommendations(movie_title, n_recommendations)
        else:
            return self._get_popularity_based_recommendations(n_recommendations)

    def _get_hybrid_recommendations(self, user_id, movie_title, n_recommendations):
        content_recs = self._get_content_based_recommendations(movie_title, n_recommendations)
        user_recs = self._get_user_based_recommendations(user_id, n_recommendations)
        
        # Combine and weight recommendations
        hybrid_scores = defaultdict(float)
        for movie, score in content_recs:
            hybrid_scores[movie] += score * 0.6
        for movie, score in user_recs:
            hybrid_scores[movie] += score * 0.4
            
        return sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

    def _get_content_based_recommendations(self, movie_title, n_recommendations):
        idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations + 1]
        return [(self.movies_df['title'].iloc[i], score) for i, score in sim_scores]

    def _get_user_based_recommendations(self, user_id, n_recommendations):
        if user_id not in self.user_profiles:
            return self._get_popularity_based_recommendations(n_recommendations)
            
        scores = []
        user_profile = self.user_profiles[user_id]
        
        for idx, row in self.movies_df.iterrows():
            if row['title'] not in self.user_ratings.get(user_id, {}):
                score = 0
                for genre in row['genre']:
                    score += user_profile['genres'].get(genre, 0)
                for keyword in row['keywords']:
                    score += user_profile['keywords'].get(keyword, 0)
                scores.append((row['title'], score))
                
        return sorted(scores, key=lambda x: x[1], reverse=True)[:n_recommendations]

    def _get_popularity_based_recommendations(self, n_recommendations):
        return [(title, pop) for title, pop in 
                zip(self.movies_df['title'], self.movies_df['popularity'])][:n_recommendations]

class RecommenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation System")
        self.root.geometry("600x400")
        self.recommender = HybridRecommender(movie_data, user_ratings)
        self.setup_gui()

    def setup_gui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(pady=10, expand=True)

        # Create tabs
        hybrid_frame = ttk.Frame(notebook)
        content_frame = ttk.Frame(notebook)
        user_frame = ttk.Frame(notebook)
        popular_frame = ttk.Frame(notebook)

        notebook.add(hybrid_frame, text='Hybrid')
        notebook.add(content_frame, text='Content-based')
        notebook.add(user_frame, text='User-based')
        notebook.add(popular_frame, text='Popular')

        # Hybrid Tab
        ttk.Label(hybrid_frame, text="User ID:").pack(pady=5)
        self.hybrid_user = ttk.Combobox(hybrid_frame, values=['user1', 'user2', 'user3', 'user4'])
        self.hybrid_user.pack(pady=5)

        ttk.Label(hybrid_frame, text="Movie Title:").pack(pady=5)
        self.hybrid_movie = ttk.Combobox(hybrid_frame, values=movie_data['title'])
        self.hybrid_movie.pack(pady=5)

        ttk.Button(hybrid_frame, text="Get Recommendations", 
                  command=self.get_hybrid_recommendations).pack(pady=10)

        # Content-based Tab
        ttk.Label(content_frame, text="Movie Title:").pack(pady=5)
        self.content_movie = ttk.Combobox(content_frame, values=movie_data['title'])
        self.content_movie.pack(pady=5)

        ttk.Button(content_frame, text="Get Recommendations", 
                  command=self.get_content_recommendations).pack(pady=10)

        # User-based Tab
        ttk.Label(user_frame, text="User ID:").pack(pady=5)
        self.user_id = ttk.Combobox(user_frame, values=['user1', 'user2', 'user3', 'user4'])
        self.user_id.pack(pady=5)

        ttk.Button(user_frame, text="Get Recommendations", 
                  command=self.get_user_recommendations).pack(pady=10)

        # Popular Tab
        ttk.Button(popular_frame, text="Get Popular Movies", 
                  command=self.get_popular_recommendations).pack(pady=10)

        # Results area for all tabs
        self.results_text = tk.Text(self.root, height=10, width=50)
        self.results_text.pack(pady=10, padx=10)

    def display_recommendations(self, recommendations, title):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"{title}\n\n")
        for movie, score in recommendations:
            self.results_text.insert(tk.END, f"- {movie}: {score:.2f}\n")

    def get_hybrid_recommendations(self):
        user_id = self.hybrid_user.get()
        movie_title = self.hybrid_movie.get()
        if not user_id or not movie_title:
            messagebox.showwarning("Input Error", "Please select both user and movie")
            return
        recommendations = self.recommender.get_recommendations(
            user_id=user_id, movie_title=movie_title)
        self.display_recommendations(recommendations, 
            f"Hybrid Recommendations for {user_id} and {movie_title}")

    def get_content_recommendations(self):
        movie_title = self.content_movie.get()
        if not movie_title:
            messagebox.showwarning("Input Error", "Please select a movie")
            return
        recommendations = self.recommender.get_recommendations(movie_title=movie_title)
        self.display_recommendations(recommendations, 
            f"Content-based Recommendations for {movie_title}")

    def get_user_recommendations(self):
        user_id = self.user_id.get()
        if not user_id:
            messagebox.showwarning("Input Error", "Please select a user")
            return
        recommendations = self.recommender.get_recommendations(user_id=user_id)
        self.display_recommendations(recommendations, 
            f"User-based Recommendations for {user_id}")

    def get_popular_recommendations(self):
        recommendations = self.recommender.get_recommendations()
        self.display_recommendations(recommendations, "Popular Movies")

def main():
    root = tk.Tk()
    app = RecommenderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()