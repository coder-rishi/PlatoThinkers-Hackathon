'''
Here we will be connecting to the model-2.pkl file,
and report as true or false
'''

# All the imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

#open model and prepare to classify
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model-2.pkl', 'rb'))
datatrue = pd.read_csv('True.csv')
datafake = pd.read_csv('Fake.csv')

#labels
datatrue['label'] = 0
datafake['label'] = 1

# concat
dataframe = pd.concat([datatrue , datafake])

# filter and sample
dataframe = dataframe[['text','label']]
dataframe = dataframe.sample(frac=1)

# prep for vectorizer
x = dataframe['text']
y = dataframe['label']

# train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)

# final function ( with some changes )
def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data=[news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

fake_news_det('trump')






