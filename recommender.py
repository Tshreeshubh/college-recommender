# recommender.py

import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load data and models
df = pd.read_csv("data/cleaned_college_data.csv")
course_matrix = joblib.load("models/course_matrix.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
le_location = joblib.load("models/location_encoder.pkl")
le_university = joblib.load("models/university_encoder.pkl")
le_ownership = joblib.load("models/ownership_encoder.pkl")

def recommend_colleges(user_course, user_location, user_university, user_ownership, top_n=5):
    # Vectorize the user course description
    user_course_vec = tfidf.transform([user_course])

    # Encode categorical preferences
    try:
        loc_enc = le_location.transform([user_location])[0]
    except:
        loc_enc = -1
    try:
        uni_enc = le_university.transform([user_university])[0]
    except:
        uni_enc = -1
    try:
        own_enc = le_ownership.transform([user_ownership])[0]
    except:
        own_enc = -1

    # Compute cosine similarity between user course and course matrix
    course_sim = cosine_similarity(user_course_vec, course_matrix).flatten()

    # Compute categorical similarity (binary match on each)
    cat_sim = (
        (df['location_encoded'] == loc_enc).astype(int) +
        (df['university_encoded'] == uni_enc).astype(int) +
        (df['ownership_encoded'] == own_enc).astype(int)
    ) / 3.0  # average categorical match

    # Final score: weighted combination
    final_score = 0.7 * course_sim + 0.3 * cat_sim

    # Get top N recommendations
    top_indices = final_score.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices][['college', 'location', 'university', 'course_offered', 'ownership_type']]
    results['score'] = final_score[top_indices].round(2)

    return results.reset_index(drop=True)

# Example usage
if __name__ == "__main__":
    # Replace with user input or connect this to a CLI or GUI later
    recommendations = recommend_colleges(
        user_course="Computer Science",
        user_location="Kathmandu",
        user_university="Tribhuvan University",
        user_ownership="private Institution",
        top_n=5
    )

    print("🎓 Top College Recommendations:")
    print(recommendations)