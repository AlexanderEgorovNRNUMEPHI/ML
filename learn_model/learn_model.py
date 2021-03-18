import os
import re
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess(text):
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()

    tokens = []
    for token in text.split():
        tokens.append(token)
    return " ".join(tokens)


def main():
    train_dir = Path(Path.cwd(), 'train')
    labels_train = []
    texts_train = []

    for label_type in ['neg', 'pos']:
        # Get the sub path
        dir_name = Path(train_dir, label_type)
        print('loading ', label_type)
        # Loop over all files in path
        for fname in tqdm(os.listdir(dir_name)):
            # Only consider text files
            if fname[-4:] == '.txt':
                # Read the text file and put it in the list
                f = open(os.path.join(dir_name, fname), encoding="utf-8")
                texts_train.append(f.read())
                f.close()
                # Attach the corresponding label
                if label_type == 'neg':
                    labels_train.append(0)
                else:
                    labels_train.append(1)

    for i, element in enumerate(texts_train):
        element = element.replace('<br />', '')
        texts_train[i] = element

    df0 = pd.DataFrame(texts_train, columns=['reviews'])
    df1 = pd.DataFrame(labels_train, columns=['labels'])

    # создали и загузили dataframe
    dataframe_train = pd.concat([df0, df1], axis=1)
    dataframe_train.to_pickle("dataframe_train")

    dataframe_train.reviews = dataframe_train.reviews.apply(lambda x: preprocess(x))
    dataframe_train.reviews.to_pickle("dataframe_train_reviews")

    vect = TfidfVectorizer(sublinear_tf=True, use_idf=True)
    x_train = vect.fit_transform(dataframe_train.reviews)
    y_train = dataframe_train.labels

    # обучили и загрузили в файл модель
    clf = LogisticRegression(solver='saga', n_jobs=-1)
    clf.fit(x_train, y_train)

    with open('clf.pkl', 'wb') as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    main()

