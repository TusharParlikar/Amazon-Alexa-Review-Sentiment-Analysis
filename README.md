# Amazon Alexa Review Sentiment Analysis

An end-to-end NLP project that classifies Amazon Alexa (Echo) product reviews as **positive** or **negative**, packaged as a Flask web app with both single-review and bulk CSV prediction.

> Based on the workflow from an end-to-end NLP walkthrough (EDA → preprocessing → model comparison → Flask deployment). See `notes.md` for a summary of that walkthrough and `project.md` for the full, commented code.

## Features

- Cleans and vectorizes raw review text (stopword removal + Porter stemming + bag-of-words).
- Compares three classifiers (Random Forest, Decision Tree, XGBoost) and ships the best one.
- A Flask app with:
  - a **single-review** predictor — type a review, get `Positive`/`Negative` back;
  - a **bulk predictor** — upload a CSV, get a prediction per row plus a pie chart of the split.
- Trained artifacts (vectorizer, scaler, model) are pickled once and reused at inference time.

## Project structure

```
amazon-alexa-sentiment/
├── data/
│   └── amazon_alexa.tsv          # not included — see "Dataset" below
├── models/
│   ├── countVectorizer.pkl       # created by train_model.py
│   ├── scaler.pkl                # created by train_model.py
│   └── model_xgb.pkl             # created by train_model.py
├── templates/
│   └── index.html
├── Output/
│   └── predictions.csv           # written after each bulk prediction
├── train_model.py                # EDA + preprocessing + training (full code in project.md)
├── app.py                        # Flask app (full code in project.md)
├── requirements.txt
└── README.md
```

## Dataset

This project is built around the public **Amazon Alexa Reviews** dataset (~3,150 reviews, tab-separated), with columns:

| Column | Meaning |
|---|---|
| `rating` | 1–5 star rating |
| `date` | review date |
| `variation` | product color/finish |
| `verified_reviews` | the review text |
| `feedback` | 1 = positive, 0 = negative (derived from `rating`) |

The dataset itself isn't bundled here — check its license/terms before redistributing it, and place your copy at `data/amazon_alexa.tsv`.

## Getting started

### 1. Set up the environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

### 2. Train the models

```bash
python train_model.py
```

This runs the EDA, cleans the text, fits the vectorizer/scaler/models, prints a train/test accuracy comparison, and pickles the vectorizer, scaler, and the chosen model into `models/`.

### 3. Run the web app

```bash
python app.py
```

Then open `http://127.0.0.1:5000/` in a browser.

## Usage

**Single review** — paste text into the box and click **Predict**; the app returns `Positive` or `Negative`.

**Bulk upload** — upload a CSV with a `verified_reviews` column (quote any review text that contains commas, as with any standard CSV); the app returns each row with a `Predicted sentiment` column added, plus a pie chart of the positive/negative split. Results are also written to `Output/predictions.csv`.

## Model performance (as reported in the walkthrough)

| Model | Train accuracy | Test accuracy |
|---|---|---|
| Random Forest (default params) | ~99–100% | ~92% |
| Decision Tree | ~99–100% | ~91% |
| **XGBoost (final model)** | ~97% | **~94%** |

⚠️ **Read this before trusting the numbers above:** the dataset is ~92% positive / 8% negative, so a model that predicts "positive" for *everything* already scores ~92% accuracy. Treat these figures as a starting point, not a finish line — see [Limitations](#limitations--next-steps). Your own numbers will vary slightly depending on the exact dataset version, random seed, and preprocessing choices.

## Limitations & next steps

- **Class imbalance isn't corrected** — accuracy is a weak metric here; precision/recall/F1 on the negative class would be more informative.
- No "neutral" class — 3★ reviews are folded into "positive."
- Hyperparameter tuning (`GridSearchCV`/cross-validation) was scoped but not completed.
- No automated tests around the Flask routes yet.

Contributions along these lines are welcome.

## Acknowledgments

Project structure and workflow adapted from a public end-to-end NLP tutorial on Amazon review sentiment analysis with a Flask front end. Code in this repo is an original reconstruction of the described workflow, not a copy of the original source.

## License

Add a license of your choice (e.g. MIT) before publishing this repository.
