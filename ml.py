import pandas as pd
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
from tensorflow.keras import models, layers, optimizers
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from enum import Enum
from log import Logger
LOG = Logger()

class Algorithms(Enum):
    DeepLearning = 1
    XGBoost = 2

class Training():
    def __init__(self,data_file:str,output_path:str,algorithm:Algorithms=Algorithms.XGBoost) -> None:
        self.algorithm=algorithm
        self.data_file=data_file
        self.output_path=output_path

    def check_requirements(self):
        LOG.INFO("Checking requirements...")
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('stopwords')

    def load(self):
        LOG.INFO("Loading data...")
        return pd.read_json(self.data_file,encoding='utf-8')

    def preprocess(self,df):
        df.drop('context',axis=1,inplace=True)
        df=df[df['patterns'].apply(lambda x:len(x))>0]
        df=df[df['responses'].apply(lambda x:len(x))>0]
        df=df[df['tag'].apply(lambda x:len(str(x).strip()))>0]
        df.reset_index(inplace=True,drop=True)
        df['tag'] = df['tag'].astype('category')
        df['class'] = df['tag'].cat.codes
        return df[['class','tag','patterns','responses']]

    def extract_features(self,df):
        wnl = WordNetLemmatizer()
        df['patterns'] = df['patterns'].apply(lambda p:[' '.join(wnl.lemmatize(w) for w in word_tokenize(s.lower()) if w.isalpha()) for s in p])
        corpus = [s for p in df['patterns'].values for s in p]
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        words = vectorizer.get_feature_names()
        result = pd.DataFrame(X.toarray(),columns=words)
        result['class_col']=-1
        classes = df['class'].values
        start=0 
        for i in range(len(df)):
            end = start+len(df.loc[i,'patterns'])
            result.loc[start:end,'class_col'] = classes[i]
            start = end 
        with open(f'{self.output_path}/words.pkl','wb') as words_file:
            pickle.dump(words,words_file)
        classes_dict = {}
        for i in df.index:
            classes_dict[df.loc[i,'class']]=df.loc[i,'tag']
        with open(f'{self.output_path}/classes.pkl','wb') as classes_file:
            pickle.dump(classes_dict,classes_file)
        return (result.drop('class_col',axis=1).to_numpy(),result['class_col'].to_numpy(),words,classes_dict)

    def build_train_model(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
        if self.algorithm==Algorithms.XGBoost:
            model = xgb.XGBClassifier()
            model.fit(X_train, y_train)
            LOG.INFO("Saving the model...")
            model.save_model(f'{self.output_path}/model.json')
            preds = model.predict(X_test)
            return (model, accuracy_score(y_test, preds))
        else:
            model = models.Sequential()
            model.add(layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(units=1, activation='softmax'))
            sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            history = model.fit(X_train, y_train, epochs=20, batch_size=5, verbose=1)
            LOG.INFO("Saving the model...")
            model.save(f'{self.output_path}/model.h5', history)
            preds = model.predict(X_test)
            return (model, accuracy_score(y_test, preds))


    def train(self):
        try:
            self.check_requirements()
            data = self.load()
            cleaned_data = self.preprocess(data)
            (X, y,words,classes) = self.extract_features(cleaned_data)
            (model,score) = self.build_train_model(X,y)
            LOG.INFO(f'model has trained with algorithm: {self.algorithm} and got score: {score}')
            return (model,words,classes)
        except Exception as exc:
            LOG.INFO('Error in training the model, check the log file for details.')
            LOG.ERROR(exc.args[0])