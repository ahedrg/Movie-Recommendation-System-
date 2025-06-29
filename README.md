#  Movie Recommendation System 

I built a fully functional **Movie Recommendation System** using a hybrid model that combines content-based, user-based, and popularity-based approaches, all packed in a clean GUI using Tkinter!

---

## Key Features

-  **Hybrid Recommendation Engine**: Combines content similarity + user preferences
-  **Content-Based Filtering**: Suggests movies similar to a selected one
- **User-Based Filtering**: Recommends movies based on user genre & keyword preferences
- **Popularity-Based**: Displays top popular movies overall
- **Graphical User Interface** using `tkinter`
- **Cosine Similarity** for calculating movie-to-movie similarity
- **MinMaxScaler** for feature normalization

---

##  How It Works

- Movies are described by features like `genre`, `keywords`, `year`, `popularity`, etc.
- Binary vectors + normalized numerical features are used to build a similarity matrix.
- Each user has a profile built from their rated movies.
- Recommendations are generated in 4 modes:
  - **Hybrid**: Combines content-based and user-based results
  - **Content-Based**: Similar movies to a selected one
  - **User-Based**: Based on user's interest profile
  - **Popular**: Top trending movies

---

## Tech Stack

- Python 3
- Pandas
- NumPy
- Scikit-learn (`MinMaxScaler`, `cosine_similarity`)
- Tkinter (GUI)

---

##  How to Run

1. Install dependencies:
```bash
pip install pandas numpy scikit-learn
