import streamlit as st    # use it for website building
import pickle   # for load file
import re   # for cleaning the data of resume
import nltk

# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(stop_words='english')

# nltk.download('punkt')
# nltk.download('stopwords')

# loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


#  function to clean the resume :
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

# web app


def main():

    nltk.download('punkt')
    nltk.download('stopwords')

    st.title("Resume Screening App")
    uplode_file = st.file_uploader(
        'upload Resume', type=['txt', 'pdf', 'docs'])

    if uplode_file is not None:

        try:
            resume_bytes = uplode_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            #    if UFT-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_feature = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_feature)[0]

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
        category_name = category_mapping.get(prediction_id, "unknown")
        st.write("Your Resume is Predicted for ---> ", category_name)


# python main
if __name__ == "__main__":
    main()
