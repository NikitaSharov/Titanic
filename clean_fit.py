import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import ensemble, linear_model, svm, neighbors
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBClassifier


features = ['Pclass', 'Fare', 'Title', 'Embarked', 'Family_type', 'Ticket_len', 'Ticket_2let', 'FareBin', 'Age']


class Models:

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.df = pd.concat([train, test], ignore_index=True)
        self.len = len(train)
        self.train = train
        self.test = test
        self.y = self.train['Survived']

    def prepare_data(self):
        self.df['Title'] = self.df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
        self.df['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)
        self.df['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
        self.df['Family_size'] = self.df['SibSp'] + self.df['Parch'] + 1
        self.df['Ticket_2let'] = self.df.Ticket.apply(lambda row: row[:2])
        self.df['Ticket_len'] = self.df.Ticket.apply(lambda row: len(row))
        self.df['Family_type'] = pd.cut(self.df['Family_size'], bins=[0, 1, 4, 7, 11], labels=['Solo', 'Small',
                                                                                               'Big', 'Very big'])

        numerical_cols = ['Fare']
        categorical_cols = ['Pclass', 'Title', 'Embarked', 'Family_type', 'Ticket_len', 'Ticket_2let', 'Age']

        # Preprocessing for numerical data
        numerical_transformer = SimpleImputer(strategy='median')

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        self.df = preprocessor.fit_transform(self.df)
        self.train = self.df[:self.len]
        self.test = self.df[self.len:]
        return self.train, self.y, self.test

    def fit(self, model, is_cross_val: bool):
        val = 0
        if is_cross_val:
            val = cross_val_score(model, self.train, self.y, cv=10).mean()
        return model.fit(self.train, self.y), val

    def choose_model(self):
        np.random.seed(42)
        alg = [
           # ensemble.AdaBoostClassifier(),
            # ensemble.BaggingClassifier(),
            # ensemble.ExtraTreesClassifier(),
            # ensemble.GradientBoostingClassifier(),
            ensemble.RandomForestClassifier(criterion='gini',
                                                       n_estimators=1750,
                                                       max_depth=7,
                                                       min_samples_split=6,
                                                       min_samples_leaf=6,
                                                       max_features='auto',
                                                       oob_score=True,
                                                       verbose=1),
            # linear_model.LogisticRegression(),
            # linear_model.PassiveAggressiveClassifier(),
            # linear_model.RidgeClassifier(),
            # linear_model.SGDClassifier(),
            # linear_model.Perceptron(),
            #
            # neighbors.KNeighborsClassifier(),
            #
            # svm.SVC(probability=True),
            # svm.NuSVC(probability=True),
            # svm.LinearSVC(),
            #
            # XGBClassifier()
        ]
        columns = ['model', 'score']
        results = pd.DataFrame(columns=columns)
        kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
        for index, model in enumerate(alg):
            model_name = model.__class__.__name__
            result = cross_val_score(model, X=self.train, y=self.y[:self.len], cv=kfolds)
            results.loc[index, 'model_parameters'] = str(model.get_params())
            results.loc[index, 'model'] = model_name
            results.loc[index, 'score'] = result.mean()
            results.loc[index, 'accuracy 3*STD'] = result.std() * 3
        results.sort_values(by='score', ascending=False).to_csv('Data/result.csv')
        best_model = results[results.score == results[['model', 'score']].max()['score']]['model'].values
        for model in alg:
            if model.__class__.__name__ == best_model:
                best_model = model
        return best_model

    def predict(self, model):
        return model.predict(self.test)






