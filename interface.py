# interface.py

from recommender import recommend_colleges

def main():
    print("\n🎓 Welcome to the College Recommendation System\n")

    # Get user inputs
    user_course = input("Enter your preferred course (e.g., Computer Science, BBA): ").strip()
    user_location = input("Enter preferred location (e.g., Kathmandu): ").strip()
    user_university = input("Enter preferred university (e.g., Tribhuvan University): ").strip()
    user_ownership = input("Enter ownership type (e.g., private Institution or community Institution): ").strip()
    top_n = input("How many recommendations do you want? (default is 5): ").strip()
    
    try:
        top_n = int(top_n)
    except:
        top_n = 5

    print("\n🔍 Finding the best matches for you...\n")

    # Get recommendations
    recommendations = recommend_colleges(
        user_course=user_course,
        user_location=user_location,
        user_university=user_university,
        user_ownership=user_ownership,
        top_n=top_n
    )

    # Display results
    if recommendations.empty:
        print("❌ No recommendations found. Please try different inputs.")
    else:
        print("✅ Recommended Colleges:\n")
        for idx, row in recommendations.iterrows():
            print(f"{idx + 1}. {row['college']} ({row['location']})")
            print(f"   University: {row['university']}")
            print(f"   Course: {row['course_offered']}")
            print(f"   Ownership: {row['ownership_type']}")
            print(f"   Match Score: {row['score']}\n")

if __name__ == "__main__":
    main()
    #python3 -m streamlit --version
    #python3 -m streamlit run "/Users/shreeshubh/suman sir assignment/models/gui_app.py"