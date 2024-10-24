import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load Data
data = pd.read_csv('Coursera.csv')

# Preprocess Course Data
data['course_content'] = data['Course Description'] + ' ' + data['Skills'] + ' ' + data['Difficulty Level']

# Vectorize Course Descriptions
vectorizer = TfidfVectorizer(stop_words='english')
course_vectors = vectorizer.fit_transform(data['course_content'])

# Save vectorizer and vectors to pickle files for faster loading in Flask
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('course_vectors.pkl', 'wb') as f:
    pickle.dump(course_vectors, f)

# Save the processed data
data.to_csv('processed_data.csv', index=False)

# Function to load model and recommend courses
def get_recommendations(feedback, top_n=5):
    # Load vectorizer and vectors
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('course_vectors.pkl', 'rb') as f:
        course_vectors = pickle.load(f)
    
    # Load processed data
    data = pd.read_csv('processed_data.csv')
    
    # Vectorize the feedback
    feedback_vector = vectorizer.transform([feedback])
    
    # Calculate similarity using cosine similarity
    cosine_sim = cosine_similarity(feedback_vector, course_vectors)
    similarity_scores = cosine_sim.flatten()
    
    # Get indices of the top_n most similar courses
    recommended_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommended_courses = data.iloc[recommended_indices]
    
    return recommended_courses[['Course Name', 'University', 'Course URL', 'Course Rating']].to_dict(orient='records')
