# Movie Recommender System

A content-based movie recommendation system built using NLP techniques and cosine similarity. The system recommends similar movies based on movie metadata including genres, keywords, cast, crew, and plot descriptions.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Algorithm](#algorithm)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Author](#author)
- [License](#license)

## Overview

This project implements a content-based filtering approach to movie recommendations. By analyzing movie metadata and computing similarity scores, the system can suggest movies that share similar characteristics with a given movie.

### Key Objectives

- Build a scalable recommendation engine using content-based filtering
- Process and vectorize movie metadata for similarity computation
- Implement cosine similarity for finding related movies
- Create an interactive interface for exploring recommendations

## Features

- **Content-Based Filtering**: Recommendations based on movie content features
- **Multi-Feature Analysis**: Combines genres, keywords, cast, crew, and overview
- **Text Preprocessing**: Stemming and stop word removal for better matching
- **Cosine Similarity**: Efficient similarity computation using vectorized operations
- **Top-K Recommendations**: Returns the most similar movies to any input

## Dataset

The system uses the **TMDB 5000 Movies Dataset**:

| Dataset | Records | Features |
|---------|---------|----------|
| tmdb_5000_movies.csv | 4,803 | 20 (budget, genres, keywords, overview, etc.) |
| tmdb_5000_credits.csv | 4,803 | 4 (cast, crew, movie_id, title) |

### Features Used for Recommendations

| Feature | Description | Processing |
|---------|-------------|------------|
| **Genres** | Movie genres (Action, Comedy, etc.) | Extracted and concatenated |
| **Keywords** | Plot keywords and themes | Extracted from JSON |
| **Cast** | Top 3 actors | Names extracted |
| **Crew** | Director | Filtered from crew data |
| **Overview** | Plot description | Tokenized and stemmed |

## Algorithm

### Content-Based Filtering Pipeline

```
Movie Metadata → Feature Extraction → Text Preprocessing → Vectorization → Similarity Matrix
                                                                              ↓
User Input → Find Movie Index → Get Similarity Scores → Sort → Top-K Recommendations
```

### Similarity Computation

The system uses **Cosine Similarity** to measure the angle between movie vectors:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Where:
- A and B are movie feature vectors
- Higher values indicate more similar movies

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/AbhinavSarkarr/Recommender-Systems.git
cd Recommender-Systems

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"
```

### Dependencies

```
pandas
numpy
scikit-learn
nltk
streamlit
pickle
```

## Usage

### Running the Preprocessing

```bash
cd "Movie Recommender System - Content Based Filtering"
jupyter notebook preprocessing.ipynb
```

### Getting Recommendations

```python
import pickle

# Load preprocessed data
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie):
    """Get top 5 similar movies"""
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommendations = []
    for i in distances[1:6]:
        recommendations.append(movies.iloc[i[0]]['title'])
    return recommendations

# Example usage
print(recommend('Spider-Man'))
# Output: ['Spider-Man 3', 'Spider-Man 2', 'The Amazing Spider-Man 2', ...]
```

### Running the Web App

```bash
streamlit run app.py
```

## How It Works

### 1. Data Preprocessing

```python
# Merge movies and credits datasets
movies = movies.merge(credits, on=['title'])

# Select relevant features
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
```

### 2. Feature Extraction

```python
# Extract names from JSON structures
def fetch_names(json_string):
    data = json.loads(json_string)
    return [item['name'] for item in data]

# Apply to genres, keywords
movies['genres'] = movies['genres'].apply(fetch_names)
movies['keywords'] = movies['keywords'].apply(fetch_names)
```

### 3. Text Processing

```python
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for word in text.split():
        if word not in stopwords.words('english'):
            y.append(ps.stem(word))
    return " ".join(y)

movies['tags'] = movies['tags'].apply(stem)
```

### 4. Vectorization

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
# Result: (4806, 5000) matrix
```

### 5. Similarity Computation

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
# Result: (4806, 4806) similarity matrix
```

## Technologies

| Technology | Purpose |
|------------|---------|
| **pandas** | Data manipulation and preprocessing |
| **NumPy** | Numerical computations |
| **scikit-learn** | Vectorization and similarity |
| **NLTK** | Text preprocessing (stemming, stopwords) |
| **Streamlit** | Web application interface |
| **pickle** | Model serialization |

## Project Structure

```
Recommender-Systems/
├── Movie Recommender System - Content Based Filtering/
│   ├── preprocessing.ipynb      # Data processing notebook
│   ├── app.py                   # Streamlit application
│   ├── movies.pkl               # Processed movie data
│   └── similarity.pkl           # Similarity matrix
├── .gitignore
└── README.md                    # This file
```

## Example Recommendations

| Input Movie | Top Recommendations |
|-------------|---------------------|
| Spider-Man | Spider-Man 3, Spider-Man 2, The Amazing Spider-Man 2 |
| Avatar | Aliens, Aliens vs Predator: Requiem, Star Trek |
| The Dark Knight | Batman Begins, The Dark Knight Rises, Batman |

## Future Enhancements

- [ ] Add collaborative filtering for hybrid recommendations
- [ ] Implement user-based personalization
- [ ] Add movie poster display in the web app
- [ ] Include rating-weighted recommendations
- [ ] Deploy on cloud platform

## Author

**Abhinav Sarkar**
- GitHub: [@AbhinavSarkarr](https://github.com/AbhinavSarkarr)
- LinkedIn: [abhinavsarkarrr](https://www.linkedin.com/in/abhinavsarkarrr)
- Portfolio: [abhinav-ai-portfolio.lovable.app](https://abhinav-ai-portfolio.lovable.app/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TMDB for the movie dataset
- scikit-learn documentation and community
- NLTK for text processing utilities

---

<p align="center">
  <strong>Discover your next favorite movie</strong>
</p>
