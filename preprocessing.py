# preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Make sure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Step 1: Load dataset
df = pd.read_csv("data/collegerecommendation.csv")

# Step 2: Drop unnecessary columns
df_cleaned = df.drop(columns=['Unnamed: 0'])

# Step 3: Standardize column names
df_cleaned.columns = [col.strip().lower().replace(" ", "_") for col in df_cleaned.columns]

# Step 4: Clean 'course_offered' field
df_cleaned['course_offered'] = df_cleaned['course_offered'].str.replace('\n', ' ', regex=True).str.strip()

# Step 5: Encode categorical features
le_university = LabelEncoder()
le_ownership = LabelEncoder()
le_location = LabelEncoder()

df_cleaned['university_encoded'] = le_university.fit_transform(df_cleaned['university'])
df_cleaned['ownership_encoded'] = le_ownership.fit_transform(df_cleaned['ownership_type'])
df_cleaned['location_encoded'] = le_location.fit_transform(df_cleaned['location'])

# Step 6: TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
course_matrix = tfidf.fit_transform(df_cleaned['course_offered'])

# Step 7: Save processed data and models
df_cleaned.to_csv("data/cleaned_college_data.csv", index=False)
joblib.dump(course_matrix, "models/course_matrix.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(le_location, "models/location_encoder.pkl")
joblib.dump(le_university, "models/university_encoder.pkl")
joblib.dump(le_ownership, "models/ownership_encoder.pkl")

print("✅ Preprocessing complete. Files saved to /data and /models.")