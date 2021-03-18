import re

def MC(text):
    import os

    from tqdm import tqdm
    train_dir = 'C:/My_Project/BC_Egorov/learn_model/train'

    labels_train = []
    texts_train = []


    for label_type in ['neg', 'pos']:
        # Get the sub path
        dir_name = os.path.join(train_dir, label_type)
        print('loading ',label_type)
        # Loop over all files in path
        for fname in tqdm(os.listdir(dir_name)):
            
            # Only consider text files
            if fname[-4:] == '.txt':
                # Read the text file and put it in the list
                f = open(os.path.join(dir_name, fname),encoding="utf-8")
                name=os.path.splitext(f.name)[0]
                texts_train.append(f.read())
                labels_train.append(name[-1])
                f.close()
              


    texts_test = text



    for i, element in enumerate(texts_train):
        element = element.replace('<br />','')
        texts_train[i] = element

    import pandas as pd
    df = pd.DataFrame(texts_train, columns =['reviews'])
    df1=pd.DataFrame(labels_train, columns =['labels'])
    dataframe_train=pd.concat([df,df1], axis=1)

    dataframe_train.labels = pd.to_numeric(dataframe_train.labels, errors='coerce')
    dataframe_train.labels.loc[dataframe_train['labels'] == 0.0] = 10.0


    dataframe_train=dataframe_train.dropna()


    dataframe_train=dataframe_train.drop_duplicates()


    dataframe_test = pd.DataFrame([texts_test], columns =['reviews'])


    from textblob import TextBlob
    def polarity_txt(text):
        return TextBlob(text).sentiment[0]


 


    def subj_txt(text):
        return  TextBlob(text).sentiment[1]



    def len_txt(text):
        return len(text)


    def count_punct(text):
        import string
        punct = string.punctuation
        count = sum([1 for char in text if char in punct])
        return round(count/(len(text) - text.count(" ")), 3)


    


    dataframe_train['lenght'] = dataframe_train['reviews'].apply(len_txt)
    dataframe_train['polarity'] = dataframe_train['reviews'].apply(polarity_txt)
    dataframe_train['subjectivity'] = dataframe_train['reviews'].apply(subj_txt)
    dataframe_train['punct%'] =  dataframe_train['reviews'].apply(lambda x: count_punct(x))
    dataframe_train.lenght=dataframe_train.lenght/dataframe_train.lenght.max()



    dataframe_test['lenght'] = len_txt(texts_test)
    dataframe_test['polarity'] = polarity_txt(texts_test)
    dataframe_test['subjectivity'] = subj_txt(texts_test)
    dataframe_test['punct%'] = count_punct(texts_test)
    dataframe_test.lenght=dataframe_test.lenght/dataframe_test.lenght.max()



    import string
    punct = string.punctuation
    import nltk
    nltk.download('wordnet')
    stopword = nltk.corpus.stopwords.words('english')
    from nltk.stem import WordNetLemmatizer


    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = "".join([word.lower() for word in text if word not in punct])
        tokens = re.split('\W+', text)
        text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopword]
        return str(text)


 


    x_train = dataframe_train[['reviews','polarity', 'subjectivity','lenght','punct%']]
    x_test = dataframe_test[['reviews','polarity', 'subjectivity','lenght','punct%']]
    y_train=dataframe_train.labels


    


    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import FeatureUnion
    from sklearn.feature_extraction import DictVectorizer
    class ItemSelector(BaseEstimator, TransformerMixin):
        def __init__(self, key):
            self.key = key

        def fit(self, x, y=None):
            return self

        def transform(self, data_dict):
            return data_dict[self.key]


    class TextStats(BaseEstimator, TransformerMixin):
        

        def fit(self, x, y=None):
            return self

        def transform(self, data):
            return [{'pos':  row['polarity'], 'sub': row['subjectivity'],'lenght': row['lenght'],'punct%': row['punct%']} for _, row in data.iterrows()]


    


    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer


    pipeline = Pipeline([
        ('union', FeatureUnion(
            transformer_list=[

                # Pipeline for pulling features from the text
                ('reviews', Pipeline([
                    ('selector', ItemSelector(key='reviews')),
                    ('tfidf', TfidfVectorizer(preprocessor=clean_text,
                         use_idf=1,smooth_idf=1)),
                ])),

                # Pipeline for pulling metadata features
                ('stats', Pipeline([
                    ('selector', ItemSelector(key=['polarity', 'subjectivity', 'lenght','punct%'])),
                    ('stats', TextStats()),  # returns a list of dicts
                    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                ])),

            ],

           
        ))
    ])


   


    pipeline.fit(x_train)


    

    train_vec = pipeline.transform(x_train)
    test_vec = pipeline.transform(x_test)


    


    from xgboost import XGBClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(with_mean=False),  XGBClassifier(eval_metric='merror',nthread=16))

    clf.fit(train_vec, y_train)

    prediction=clf.predict(test_vec)
    
    return prediction
