import re
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess(text):
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()

    tokens = []
    for token in text.split():
        tokens.append(token)
    return " ".join(tokens)


def result_of_bc(text):
    dataframe_train = pd.read_pickle("dataframe_train")
    dataframe_train.reviews = pd.read_pickle("dataframe_train_reviews")

    with open('clf.pkl', 'rb') as f:
        new_clf = pickle.load(f)

    texts_test = text
    vect = TfidfVectorizer(sublinear_tf=True, use_idf=True)
    vect.fit_transform(dataframe_train.reviews)
    review_text = preprocess(texts_test)
    review_text = [review_text]
    x_test = vect.transform(review_text)
    y_pred = new_clf.predict(x_test)
    return y_pred