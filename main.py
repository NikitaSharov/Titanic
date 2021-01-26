from clean_fit import Models
import pandas as pd
import os


def main():
    train = pd.read_csv('Data/train.csv')
    test = pd.read_csv('Data/test.csv')
    clf = Models(train, test)
    clf.prepare_data()
    m = clf.choose_model()
    model, score = clf.fit(model=m, is_cross_val=True)
    print('Submission score : {0:2f} using model {1}'.format(score, str(model.__class__.__name__)))
    predictions = clf.predict(model)
    output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)
    os.system('kaggle competitions submit -f submission.csv -m "submission using kaggle API" -q titanic')


if __name__ == main():
    main()
