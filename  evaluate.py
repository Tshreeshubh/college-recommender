# evaluate.py

from recommender import recommend_colleges

# Define a few test cases
test_cases = [
    {
        "course": "Computer Science",
        "location": "Kathmandu",
        "university": "Tribhuvan University",
        "ownership": "private Institution"
    },
    {
        "course": "Information Technology",
        "location": "Lalitpur",
        "university": "Tribhuvan University",
        "ownership": "community Institution"
    },
    {
        "course": "BBA",
        "location": "Bhaktapur",
        "university": "Westcliff University, CA, USA",
        "ownership": "private Institution"
    }
]

def run_evaluation():
    print("🧪 Running Evaluation...\n")
    for i, case in enumerate(test_cases):
        print(f"🔍 Test Case #{i+1}:")
        print(f"Course: {case['course']}")
        print(f"Location: {case['location']}")
        print(f"University: {case['university']}")
        print(f"Ownership: {case['ownership']}")
        
        results = recommend_colleges(
            user_course=case['course'],
            user_location=case['location'],
            user_university=case['university'],
            user_ownership=case['ownership'],
            top_n=3
        )

        for idx, row in results.iterrows():
            print(f"\n {idx+1}. {row['college']} | {row['location']} | Score: {row['score']}")
            print(f"    Course: {row['course_offered']}")
            print(f"    University: {row['university']}")
            print(f"    Ownership: {row['ownership_type']}")
        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    run_evaluation()