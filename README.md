# A Hybrid Movie Recommender: Collaborative and Content-Based Filtering

A movie recommender system built from first principles, combining item-item collaborative filtering and genre-based content filtering into a single hybrid model, evaluated against baselines on the MovieLens dataset.

## Overview

This project recommends movies using the MovieLens dataset (100,836 ratings from 610 users across 9,724 movies), a standard real-world benchmark for recommendation. It implements two complementary approaches from scratch and blends them:

1. Collaborative filtering, which recommends based on the rating patterns of similar items, using item-item cosine similarity.
2. Content-based filtering, which recommends movies similar in genre to those a user already rates highly.
3. A hybrid model that blends the two with a tunable weight, validated by held-out RMSE against baselines.
4. A neural collaborative filtering model in Keras that learns user and movie embeddings, compared against the classical methods.

Every rated movie in the dataset also carries genre metadata, so the ratings and content can be joined cleanly on a shared movie identifier. Both data files ship with the repository, so the notebook runs without any network access.

## Selected results

Models were evaluated by root mean squared error (RMSE) on held-out ratings, on the 0.5 to 5.0 rating scale. Lower is better.

| Model | RMSE |
|-------|------|
| Global mean (baseline) | 1.06 |
| User mean (baseline) | 0.95 |
| Content-based | 1.00 |
| Collaborative filtering | 0.87 |
| Hybrid (alpha = 0.7) | 0.88 |
| Neural collaborative filtering | 0.85 |

Among the classical methods, collaborative filtering is the strongest and clearly beats both baselines, while blending in the genre-based content score does not improve accuracy further. This is a useful finding in itself: a hybrid is only worthwhile when each component contributes information the other lacks, and here the content signal is largely redundant for rating prediction. The neural collaborative filtering model, which learns user and movie embeddings directly from the ratings, achieves the best accuracy overall, though by a modest margin. This is consistent with the general pattern that classical similarity methods remain strong baselines on small, sparse datasets, with neural approaches pulling further ahead at larger scale. The matrix is over 98 percent empty, which is the central difficulty of recommendation: the model must infer the unrated majority of entries from the rated minority.

## How to run

```bash
git clone https://github.com/ekahorsu/movie-recommender-hybrid.git
cd movie-recommender-hybrid
pip install -r requirements.txt
jupyter notebook movie_recommender_hybrid.ipynb
```

Tested with Python 3.10. No internet connection is required; both data files are included.

## Tech stack

- **NumPy** for the from-scratch similarity and prediction computations
- **pandas** for the user-item matrix and data handling
- **Matplotlib** for the data-exploration and hybrid-weighting plots
- **scikit-learn** for the train/test split and RMSE metric
- **TensorFlow / Keras** for the neural collaborative filtering model

## Repository contents

- `movie_recommender_hybrid.ipynb` - the full analysis notebook
- `ratings.csv` - 100,836 user ratings
- `movies.csv` - movie titles and genres
- `requirements.txt` - dependencies

## Data source

The data is the MovieLens latest-small dataset, collected and distributed by GroupLens Research at the University of Minnesota. It contains ratings and genre metadata for use in recommendation research and education.

Original source: https://grouplens.org/datasets/movielens/

The copy included in this repository is provided for convenience and reproducibility, so that the notebook runs without any network access.

## Challenges and Solutions

### Joining ratings to content metadata

**Challenge.** A hybrid recommender needs both user ratings, for collaborative filtering, and item attributes, for content-based filtering. These two kinds of information often come from separate sources that use different movie identifiers and only partially overlap, which makes a reliable join difficult.

**Solution.** The MovieLens dataset was used in a form where the ratings and the movie metadata share a common movie identifier, giving complete alignment between the two: every rated movie also has genre information. This removed the need for fragile title-based matching and allowed the collaborative and content signals to be combined on exactly the same set of movies.

### Sparsity of the user-item matrix

**Challenge.** More than 98 percent of the user-item matrix is empty, because each user rates only a small fraction of all movies. Similarities computed from so few co-ratings can be unreliable, and unrated movies carry no collaborative signal at all.

**Solution.** Ratings were mean-centered per user before computing cosine similarity, so that differences in individual rating habits do not distort the similarity measure, and predictions were limited to the most similar already-rated items. The content-based component provides a fallback signal for cases where collaborative information is thin.

### Reporting performance honestly against baselines

**Challenge.** A recommender can appear to perform well while doing little better than predicting an average rating, so a single accuracy figure can be misleading.

**Solution.** Every model was compared against two baselines, the global mean rating and each user's own mean rating. This shows how much of the accuracy comes from the modeling itself rather than from guessing an average, and it makes clear that collaborative filtering provides a genuine improvement while the hybrid blend, on this dataset, does not add further benefit over collaborative filtering alone.
