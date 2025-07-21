import pandas as pd

# Load your existing dataset
df = pd.read_csv("/Users/shreeshubh/suman sir assignment/cleaned_college_data.csv")

# New college entry in correct order
new_college = {
    "college": "Arniko International Secondary School",
    "location": "Talchikhel, Lalitpur",
    "university": "NEB",
    "course_offered": "+2 (Science, Management)",
    "ownership_type": "Private",
    "phone_number": "01_5571341",
    "email": "bhandari@gmail.com",
    "university_encoded": 34,
    "ownership_encoded": 1,
    "location_encoded": 299
}

# Append new data
df = pd.concat([df, pd.DataFrame([new_college])], ignore_index=True)

# Save updated dataset
df.to_csv("/Users/shreeshubh/suman sir assignment/updated_college_data.csv", index=False)

print("✔ Arniko International Secondary School added successfully.")