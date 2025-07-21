# visualize.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Create folder for saving visualizations
os.makedirs("visuals", exist_ok=True)

# Load cleaned dataset and models with correct path
try:
    df = pd.read_csv("/Users/shreeshubh/suman sir assignment/data/cleaned_college_data.csv")
    print("✅ Dataset loaded successfully")
except FileNotFoundError:
    print("❌ Dataset not found. Please check the file path.")
    exit()

try:
    tfidf = joblib.load("/Users/shreeshubh/suman sir assignment/models/tfidf_vectorizer.pkl")
    course_matrix = joblib.load("/Users/shreeshubh/suman sir assignment/models/course_matrix.pkl")
    print("✅ Models loaded successfully")
except FileNotFoundError:
    print("❌ Models not found. Please check if you have run the training script first.")
    exit()

# Set Seaborn style
sns.set_style("whitegrid")

print("📊 Creating visualizations...")

# 1. 📊 Bar Chart: Colleges per Location
plt.figure(figsize=(12, 6))
location_counts = df['location'].value_counts()
bars = plt.bar(range(len(location_counts)), location_counts.values, color='skyblue', alpha=0.7)
plt.title("Number of Colleges by Location", fontsize=16, fontweight='bold')
plt.xlabel("Location", fontsize=12)
plt.ylabel("Number of Colleges", fontsize=12)
plt.xticks(range(len(location_counts)), location_counts.index, rotation=45, ha='right')

# Add value labels on bars
for bar, value in zip(bars, location_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             str(value), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("visuals/bar_location_count.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Location bar chart saved")

# 2. 🥧 Pie Chart: Ownership Type Distribution
plt.figure(figsize=(8, 8))
ownership_counts = df['ownership_type'].value_counts()
colors = ['lightgreen', 'salmon', 'lightblue', 'orange', 'purple'][:len(ownership_counts)]

plt.pie(ownership_counts.values, labels=ownership_counts.index, autopct='%1.1f%%',
        colors=colors, wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
        startangle=90, textprops={'fontsize': 12})
plt.title("Ownership Type Distribution", fontsize=16, fontweight='bold')
plt.axis('equal')
plt.savefig("visuals/pie_ownership.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Ownership pie chart saved")

# 3. 🌡️ Heatmap: Encoded Feature Correlation
plt.figure(figsize=(10, 8))
# Check which encoded columns exist
encoded_cols = []
for col in ['university_encoded', 'ownership_encoded', 'location_encoded']:
    if col in df.columns:
        encoded_cols.append(col)

if encoded_cols:
    correlation_matrix = df[encoded_cols].corr()
    mask = correlation_matrix.isnull()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                linewidths=0.5, square=True, mask=mask,
                cbar_kws={'shrink': 0.8}, fmt='.3f')
    plt.title("Correlation Heatmap of Encoded Features", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("visuals/heatmap_encoded_features.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Correlation heatmap saved")
else:
    print("⚠️ No encoded columns found, skipping correlation heatmap")

# 4. ☁️ WordCloud: Most Common Words in Course Offerings
if 'course_offered' in df.columns:
    # Clean and prepare text data
    course_text = df['course_offered'].dropna().astype(str)
    all_text = ' '.join(course_text.values)
    
    # Remove common stop words and clean text
    from collections import Counter
    import re
    
    # Basic text cleaning
    all_text = re.sub(r'[^\w\s]', ' ', all_text.lower())
    all_text = ' '.join(all_text.split())  # Remove extra whitespace
    
    if all_text.strip():
        wordcloud = WordCloud(width=1200, height=600, 
                            background_color='white', 
                            colormap='viridis',
                            max_words=100,
                            relative_scaling=0.5,
                            random_state=42).generate(all_text)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("WordCloud of Courses Offered", fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig("visuals/wordcloud_courses.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ WordCloud saved")
    else:
        print("⚠️ No course text data available for WordCloud")
else:
    print("⚠️ 'course_offered' column not found, skipping WordCloud")

# 5. 📈 Histogram: Cosine Similarity Scores for a Sample Input
sample_course = "Computer Science"
try:
    sample_vector = tfidf.transform([sample_course])
    cos_similarities = cosine_similarity(sample_vector, course_matrix).flatten()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(cos_similarities, bins=30, kde=True, color='purple', alpha=0.7)
    plt.axvline(cos_similarities.mean(), color='red', linestyle='--', 
                label=f'Mean: {cos_similarities.mean():.3f}')
    plt.title(f"Cosine Similarity Distribution: '{sample_course}' vs All Courses", 
              fontsize=16, fontweight='bold')
    plt.xlabel("Cosine Similarity Score", fontsize=12)
    plt.ylabel("Number of Colleges", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visuals/hist_cosine_similarity.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Cosine similarity histogram saved")
except Exception as e:
    print(f"⚠️ Could not create similarity histogram: {e}")

# 6. Additional: College Count by University Type (if available)
if 'university' in df.columns:
    plt.figure(figsize=(12, 6))
    university_counts = df['university'].value_counts().head(10)  # Top 10
    bars = plt.bar(range(len(university_counts)), university_counts.values, 
                   color='lightcoral', alpha=0.7)
    plt.title("Top 10 Universities by College Count", fontsize=16, fontweight='bold')
    plt.xlabel("University", fontsize=12)
    plt.ylabel("Number of Colleges", fontsize=12)
    plt.xticks(range(len(university_counts)), university_counts.index, 
               rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars, university_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 str(value), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("visuals/bar_university_count.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ University bar chart saved")

# Display summary
print("\n" + "="*50)
print("📊 VISUALIZATION SUMMARY")
print("="*50)
print(f"📁 Dataset shape: {df.shape}")
print(f"📂 Visualizations saved in: {os.path.abspath('visuals/')}")

# List all created files
visual_files = [f for f in os.listdir('visuals') if f.endswith('.png')]
for i, file in enumerate(visual_files, 1):
    print(f"  {i}. {file}")

print("\n✅ All visualizations have been created successfully!")
print("🎨 You can now view the charts in the /visuals folder.")