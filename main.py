# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
            
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# %%
import pandas as pd
true_df = pd.read_csv('extension/True.csv')
fake_df = pd.read_csv('extension/Fake.csv')


# %%
true_df['label'] = 0
fake_df['label'] = 1


# %%
dataset = pd.concat([true_df , fake_df])


# %%
dataset = dataset[['text','label']]


# %%
dataset.head()


# %%
dataset = dataset.sample(frac=1)


# %%
dataset.head()


# %%
x = dataset['text']
y = dataset['label']

# %%
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# %%
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)


# %%
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)


# %%
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfid_x_train, y_train)


# %%
y_pred = classifier.predict(tfid_x_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100, 9)}%')


# %%
cf = confusion_matrix(y_test, y_pred, labels =[1, 0])
print(cf)


# %%
def fake_news_det(news):
    input_data=[news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = classifier.predict(vectorized_input_data)
    print(prediction)


# %%
import pickle
pickle.dump(classifier, open('model.pkl', 'wb'))

fake_news_det('Biden has planned to withdraw all U.S. troops from Afghanistan.')
# that's it, guys! we now have a model.pkl file.


# %%



