from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re

df = pd.read_csv('UpdatedResumeDataSet.csv')

# Bar chart and Pie chart for visulization:

# (df.head())
# (df.shape)
# (df.columns)
# (df['Category'].value_counts())
# plt.figure(figsize=(15,5))
# sns.countplot(df['Category'])
# plt.xticks(rotation=90)
# plt.show()
# counts = df['Category'].value_counts()
# labels =df['Category'].unique()
# plt.figure(figsize=(15,10))
# plt.pie(counts,labels=labels,autopct='%1.1f%%',shadow=True)
# plt.show()

# Function to remove special characters and unnecessary text from resume:


def clean_resume(text):
    if not isinstance(text, str):
        return ""

    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Remove non-alphanumeric characters (except space)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Replace multiple whitespace characters with single space
    text = re.sub(r'\s+', ' ', text)

    # Lowercase everything
    text = text.lower()

    return text.strip()


df['Resume'] = df['Resume'].apply(clean_resume)
# print(df['Resume'])


le = LabelEncoder()
le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])
# print(df.Category.unique())


#  ['Data Science' 'HR' 'Advocate' 'Arts' 'Web Designing'
#  'Mechanical Engineer' 'Sales' 'Health and fitness' 'Civil Engineer'
#  'Java Developer' 'Business Analyst' 'SAP Developer' 'Automation Testing'
#  'Electrical Engineering' 'Operations Manager' 'Python Developer'
#  'DevOps Engineer' 'Network Security Engineer' 'PMO' 'Database' 'Hadoop'
#  'ETL Developer' 'DotNet Developer' 'Blockchain' 'Testing']


#  vectorization

tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(df['Resume'])
requiredtxt = tfidf.transform(df['Resume'])
# print(requiredtxt)


# spiliting and train the model with dataset in train and test.Here 80% of data is used for train and 20% is for test.

x_train, x_test, y_train, y_test = train_test_split(
    requiredtxt, df['Category'], test_size=0.2, random_state=42)
# print(x_test.shape)
# print(x_train.shape)

# it is a multiple clasification , so here we use k neighbors algo bcz it uses the nearest point.

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
# print(accuracy_score(y_test, pred))

pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))


# prediction System

myresume = """John Doe
123 Anywhere st. New York, NY
Phone: 555-555-1212
email: johndoe@email
linkedin.com/in/johndoe123
github.com/johndoe122

Objective
Looking for job. I like computers and work. Hope to get good position somewhere.

Experience

software enginer
Googl, Inc.
2019/5 - Present

writ code

team stuff

some meetings

worked on cloud maybe

debugging sometimes

Software developer
startUP company
Aug 2016 - 2018

making apps (2)

doing backend and frontend, i guess

sometimes talk to clients

fix server things (mysql, etc.)

McDonalds
Cashier and kitchen
2014-2015

gave people food

multitask

Educatoin
BS, computer
Big State Universty, 2012-2016
took classes in CS and IT. Some GPA, didn't fail

Skills

python, c++, js, HTML, etc.

linux and stuff

teamplay

agile maybe

fast typist

powerpoint (kinda)

Certificatons
AWS maybe?
Scrum one idk

Projects

built some websites

made a todo app in javascript once

raspberry pi robot w/ camera

"""

# load the trained classifier
clf = pickle.load(open('clf.pkl', 'rb'))

# clean the input resume
cleaned_resume = clean_resume(myresume)

# transform the cleaned resume using tfidfvectorizer
input_features = tfidf.transform([cleaned_resume])

# making the prediction using the loaded classifier
predition_id = clf.predict(input_features)[0]

# map category id to category name
category_mapping = {
    6: "Data Science",
    12: "HR",
    0: 'Advocate',
    1: 'Arts',
    24: 'Web Designing',
    16: 'Mechanical Engineer',
    22: 'Sales',
    14: 'Health and fitness',
    5:  'Civil Engineer',
    15: 'Java Developer',
    4: 'Business Analyst',
    21: 'SAP Developer',
    2: 'Automation Testing',
    11: 'Electrical Engineering',
    18: 'Operations Manager',
    20: 'Python Developer',
    8: 'DevOps Engineer',
    17: 'Network Security Engineer',
    19: 'PMO',
    7: 'Database',
    13: 'Hadoop',
    10: 'ETL Developer',
    9:  'DotNet Developer',
    3: 'Blockchain',
    23: "Testing",
}
category_name = category_mapping.get(predition_id, "unknown")
print("Your Resume is Predicted for ---> ", category_name)

