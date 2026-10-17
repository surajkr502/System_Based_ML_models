#  Movie Recommendation System

A content-based movie recommender built on the TMDB 5000 Movie Dataset. Give it a movie title, and it returns the 5 most similar movies based on genre, keywords, cast, director, and plot overview.

 **Live Demo:** [movierecommederapp.streamlit.app](https://movierecommederapp.streamlit.app/)

---

##  Table of Contents

- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Recommendation Logic](#recommendation-logic)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Dataset

Source: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle)

The project uses two files because they hold different kinds of information:

- **`tmdb_5000_movies.csv`** has the movie metadata: `id`, `title`, `overview`, `genres`, `keywords`, `budget`, `revenue`, and more.
- **`tmdb_5000_credits.csv`** has the people involved: `movie_id`, `title`, `cast`, and `crew`.

To use the cast and crew with the movie metadata, the two files have to be merged.

### Why merge on `movie_id` and not `title`?

Titles are not unique. The dataset has duplicate titles, and two different movies can share a name. So merging on `title` could join the wrong rows. Each movie has a unique ID, so merging on that is safe.

The two files name this column differently (`id` in movies, `movie_id` in credits). I renamed `id` to `movie_id` before merging. After the merge, the shape is **4803 × 7** (only the columns needed for the model are kept).

### Why duplicate titles are kept

Duplicate titles are not removed, because two different movies can have the same name. Instead, when a title lookup is needed, the first match is used (see [Recommendation Logic](#recommendation-logic)).

---

## Pipeline

### 1. Parsing the string columns with `ast`

`genres`, `keywords`, `cast`, and `crew` are stored as strings that look like lists of dictionaries:

```python
'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
```

Python can't work with these directly, so `ast.literal_eval` converts each string into a real list of dictionaries. Then the values are pulled out:

| Column | What is kept |
|---|---|
| `genres` | All genre names |
| `keywords` | All keyword names |
| `cast` | Names of the first 3 actors |
| `crew` | Name of the director only |

### 2. Extracting the director

The `crew` column stores every job role (director, producer, composer, and so on). Only the entry where `job == 'Director'` is kept.

### 3. Turning the overview into a list of words

The `overview` column is a long string. A lambda function splits it on spaces and returns a list of words:

```python
movies['overview'] = movies['overview'].apply(lambda x: x.split())
```

### 4. Removing spaces inside names

A name such as `James Cameron` would otherwise become two tokens, `james` and `cameron`. Any match on `james` would then count, even for a different person. To prevent this, the spaces inside each name are removed:

```python
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(' ', '') for i in x])
# James Cameron -> JamesCameron (one token)
```

This also fixes a tokenization problem, because the whole name is now treated as one word.

### 5. Weighting the features

The lists are combined into one `tags` column. Two columns are repeated to give them more weight:

- **Director** is repeated **3 times**, so director-based movies rank higher.
- **Genres** are repeated **2 times**.

```python
movies['tags'] = (
    movies['keywords'] +
    movies['crew'] * 3 +
    movies['cast'] +
    movies['genres'] * 2 +
    movies['overview']
)
```

### 6. Selecting the columns for the model

```python
new_df = movies[['movie_id', 'title', 'tags']].copy().reset_index(drop=True)
```

A note on brackets: `df['col']` returns a column (a Series), while `df[['col']]` returns a DataFrame. The double bracket is needed to select several columns.

### 7. Converting lists to strings

`CountVectorizer` and `TfidfVectorizer` understand text, not lists. So each list is joined into one string, with words separated by spaces:

```python
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
```

### 8. Lowercasing

```python
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
```

Lowercasing makes `Action` and `action` the same token. Without it, the same word written with different capitals would be counted as two different words, which lowers the similarity between movies that should match.

### 9. Stemming

Words are reduced to their base form with the Porter stemmer (for example, `loved` and `loving` both become `love`). Stemming is applied to the `tags` column before vectorization, so the vectorizers see the stemmed text.

### 10. Vectorization

The `tags` column is stored as `corpus`. Two vectorizers are built on the same corpus:

```python
cv = CountVectorizer(max_features=5000, stop_words='english')
cv_vectors = cv.fit_transform(corpus)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english', sublinear_tf=True)
tfidf_vectors = tfidf.fit_transform(corpus)
```

- `fit` learns the vocabulary from the corpus.
- `transform` turns each movie into a vector of word counts (or TF-IDF weights).
- `stop_words='english'` removes common words such as "the" and "and".
- `max_features=5000` keeps the 5000 most frequent words.

After fitting, the shape of each matrix is checked. Both should have the same number of columns and one row per movie.

---

## Recommendation Logic

### Title-to-position lookup

Each movie is identified by its **position** (index), starting from 0. A lookup table maps each title to its position. If a title appears more than once, only the first match is kept, so every title gives one position.

```python
title_to_idx = pd.Series(np.arange(len(new_df)), index=new_df['title'].values)
title_to_idx = title_to_idx[~title_to_idx.index.duplicated()]
```

### Top-5 function

```python
def top_k(vectors, idx, k=5):
    scores = cosine_similarity(vectors[idx], vectors).ravel()
    scores[idx] = -1                                  # exclude the movie itself
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])]
```

For a selected movie, the function compares its vector with every other vector using cosine similarity. It returns the **positions** of the 5 most similar movies, sorted from most to least similar. Only one row is computed at a time, so the full 4803 × 4803 matrix is never built.

### Genre comparison with Jaccard similarity

To check whether recommendations are relevant, a genre comparison is used. The genres are taken from the original `movies` dataframe and stored in a dictionary:

```python
genre_map = dict(zip(movies['movie_id'], movies['genres']))
```

Two movies are compared with **Jaccard similarity**:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

This is the number of genres the two movies share, divided by the total number of different genres across both. For example:

- Movie A: {Action, Science Fiction}
- Movie B: {Action, Drama}
- Common: {Action} → 1
- All genres: {Action, Science Fiction, Drama} → 3
- Score: 1 / 3 ≈ **0.33**

A score of 1.0 means identical genres, and 0.0 means no genre in common.

---

## Evaluation

Five test movies were chosen to cover different cases:

| Movie | What it tests |
|---|---|
| Avatar | Popular sci-fi blockbuster |
| The Dark Knight Rises | Franchise and director (Nolan / Batman) |
| Silent Running | Obscure sci-fi film (hardest case) |
| Spectre | Franchise and cast |
| Toy Story | Non-sci-fi genres (animation, family) |

For each movie, the top 5 recommendations are returned by both models, and their average genre overlap is calculated. The results are printed movie by movie, and the average is taken for each model.

| Model | Avg. genre overlap |
|---|---|
| CountVectorizer | 0.729 |
| TF-IDF | 0.466 |

**How to read this:** Genre overlap is a proxy metric. Genres are part of the input, so the metric favours models that weight raw genre words heavily. TF-IDF lowers the weight of common words and gives more weight to rare signals such as director and specific keywords, which this metric does not measure. Qualitative checks support this: Avatar returns James Cameron films, and The Dark Knight Rises returns Batman films. A manual relevance rating is the recommended next step before picking a final model.

---

## Project Structure

```
.
├── Movies_Recomm_system.ipynb    # Full pipeline: cleaning, features, models, evaluation, export
├── tmdb_5000_movies.csv          # Raw movie metadata
├── tmdb_5000_credits.csv         # Raw cast and crew data
├── movie_recommender.pkl         # Saved artifacts for the app (generated by the notebook)
├── requirements.txt              # Python dependencies
├── app.py                        # Streamlit app
└── README.md
```

### Saved artifacts

Instead of saving the full similarity matrix (about 185 MB), the notebook saves:

- The `movie_id` and `title` table
- A precomputed top-20 neighbours table (4803 × 20 integers)
- The fitted vectorizer and model name

All of this goes into one file, `movie_recommender.pkl`.

---

## Installation

**Requirements:** Python 3.10 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

To run the notebook from scratch, also place the two TMDB CSV files in the project folder.

---

## Usage

### Use the saved artifacts in Python

```python
import pickle
import numpy as np
import pandas as pd

with open('movie_recommender.pkl', 'rb') as f:
    art = pickle.load(f)

movies = art['movies']
neighbors = art['neighbors']

title_to_idx = pd.Series(np.arange(len(movies)), index=movies['title'].values)
title_to_idx = title_to_idx[~title_to_idx.index.duplicated()]

def recommend(title, k=5):
    idx = title_to_idx[title]
    return movies['title'].iloc[neighbors[idx][:k]].tolist()

print(recommend('Avatar'))
# ['Aliens', 'Terminator 2: Judgment Day', 'The Terminator', 'True Lies', 'The Abyss']
```

### Re-run the notebook

Open `Movies_Recomm_system.ipynb` in Jupyter and run the cells in order. The order matters: `new_df` must be created with a reset index before the vectorizers and helper functions are built.

---

## Limitations

- **Director bias:** the director weight pushes results toward other films by the same director.
- **Name collisions:** two different people with the same name are treated as one.
- **Duplicate titles:** when two movies share a title, only the first one can be found by name.
- **No user data:** this is content-based only. It does not use user ratings or watch history.
- **Small evaluation:** the comparison uses 5 test movies and genre overlap as the metric.
- **Missing posters:** the dataset has no poster images. These need the TMDB API.

---

## Future Improvements

- [ ] Manual relevance rating for both models
- [x] Streamlit app with a movie dropdown and cached artifact loading (live now)
- [ ] Sentence embeddings (e.g., `all-MiniLM-L6-v2`) for better plot-level similarity
- [ ] Hybrid model combining content similarity with user ratings
- [ ] Movie posters via the TMDB API
- [ ] Unit tests for the cleaning and recommendation functions

---

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (CountVectorizer, TfidfVectorizer, cosine similarity)
- NLTK (Porter stemmer)
- Streamlit (deployed on Streamlit Community Cloud)

---

## License

This project is for learning and portfolio purposes. The dataset is subject to its own license on Kaggle. Please check the dataset page before redistribution.

---

## Acknowledgements

- [TMDB](https://www.themoviedb.org/) for the dataset.
- scikit-learn and pandas documentation.
