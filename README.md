# Fake News Detector

A simple Python project that trains a logistic regression model to classify news articles as real or fake using TF-IDF text features.

## Project structure

- `fake_news_detector.py` - main script that loads data, cleans text, trains the model, evaluates accuracy, and shows a confusion matrix.
- `fake_or_real_news.csv` - dataset used to train and test the model.
- `requirements.txt` - Python package dependencies.

## Requirements

- Python 3.8+
- pandas
- numpy
- nltk
- scikit-learn
- matplotlib
- seaborn

## Setup

1. Create and activate a Python environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK stopwords (the script does this automatically on first run):

```bash
python fake_news_detector.py
```

## Usage

Run the project with:

```bash
python fake_news_detector.py
```

The script will:

- load `fake_or_real_news.csv`
- clean and preprocess text
- vectorize text using TF-IDF
- train a logistic regression classifier
- evaluate accuracy on a test split
- display a confusion matrix
- print an example prediction

## Notes

- The model predicts `Real` for label `1` and `Fake` for label `0`.
- Customize the `example` text in `fake_news_detector.py` to test new inputs.

## License

This project is provided as-is for educational use.