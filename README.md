# 🎓 Course-Based College Recommendation System

An intelligent college recommendation system that helps students find the perfect college match based on their course preferences, location, university affiliation, and institution type preferences.
## 🏗️ System Architecture

```
📁 college-recommender/
├── 📊 data/
│   └── cleaned_college_data.csv
├── 🤖 models/
│   ├── course_matrix.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── location_encoder.pkl
│   ├── university_encoder.pkl
│   └── ownership_encoder.pkl
├── 🎨 visuals/
│   └── (generated visualizations)
├── 📝 Core Files
│   ├── recommender.py          # Main recommendation engine
│   ├── gui_app.py             # Streamlit web interface
│   ├── interface.py           # Command-line interface
│   ├── evaluate.py            # Performance evaluation
│   ├── visualize.py           # Data visualization
│   └── preprocessing.py       # Data cleaning utilities
└── 📋 README.md
```
## 📊 System Performance

- **Precision@5**: 92.0% 
- **Top-5 Accuracy**: 100%
- **Mean Reciprocal Rank**: 100%
- **NDCG@5**: 94.4%

*Performance Grade: **EXCELLENT** ⭐⭐⭐⭐⭐*

## 🌟 Features

- **Multi-criteria Recommendation**: Considers course, location, university, and ownership type
- **Fuzzy String Matching**: Handles typos and variations in user input
- **TF-IDF Vectorization**: Advanced text similarity for course matching
- **Adaptive Weighting**: Dynamic weight adjustment based on input specificity
- **Multiple Interfaces**: CLI, GUI (Streamlit), and programmatic API
- **Comprehensive Evaluation**: Built-in evaluation metrics and visualization tools



## 🚀 Quick Start

### Prerequisites

```bash
# Required Python packages
pip install pandas numpy scikit-learn streamlit matplotlib seaborn wordcloud joblib
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/college-recommender.git
cd college-recommender
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare the data**
```bash
# Ensure your dataset is at: data/cleaned_college_data.csv
# The system will automatically create model files on first run
```

## 💻 Usage

### 1. 🌐 Web Interface (Recommended)

Launch the interactive Streamlit web application:

```bash
streamlit run gui_app.py
```

**Features:**
- ✨ Modern, intuitive interface
- 📋 Dropdown selections with custom input options
- 📊 Real-time recommendations with similarity scores
- 💾 Download results as CSV
- 📈 Detailed data tables

### 2. 🖥️ Command Line Interface

For terminal-based interaction:

```bash
python interface.py
```

**Features:**
- 🎯 Interactive prompts for user preferences
- 💡 Built-in help and examples
- 🔄 Multiple search sessions
- 📋 Formatted recommendation display

### 3. 🔧 Programmatic API

Use the recommendation engine in your code:

```python
from recommender import recommend_colleges

# Get recommendations
results = recommend_colleges(
    user_course="Computer Science",
    user_location="Kathmandu", 
    user_university="Tribhuvan University",
    user_ownership="Private Institution",
    top_n=10
)

print(results)
```

## 📊 Evaluation & Performance

### Run Comprehensive Evaluation

```bash
python evaluate.py
```

**Evaluation Metrics:**
- **Precision@K**: Measures accuracy of top-K recommendations
- **Top-N Accuracy**: Whether relevant items appear in top-N results
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks
- **NDCG**: Normalized Discounted Cumulative Gain

### Current Performance Results

| Metric | Score |
|--------|-------|
| Precision@1 | 100% |
| Precision@3 | 93.3% |
| Precision@5 | 92.0% |
| Top-5 Accuracy | 100% |
| MRR | 100% |

## 📈 Visualizations

Generate insightful visualizations:

```bash
python visualize.py
```

**Generated Charts:**
- 📊 College distribution by location
- 🥧 Ownership type distribution  
- 🌡️ Feature correlation heatmaps
- ☁️ Course offerings word cloud
- 📈 Similarity score distributions

## 🛠️ Technical Details

### Algorithm Components

1. **Text Processing**
   - TF-IDF vectorization for course similarity
   - Fuzzy string matching with SequenceMatcher
   - Course abbreviation expansion (CS → Computer Science)

2. **Similarity Calculations**
   - **Course Similarity**: Cosine similarity on TF-IDF vectors
   - **Location Similarity**: Fuzzy matching with geographic alternatives
   - **University Similarity**: Word overlap + fuzzy matching
   - **Ownership Similarity**: Normalized categorical matching

3. **Scoring & Ranking**
   - Adaptive weight calculation based on input specificity
   - Multi-criteria score aggregation
   - Minimum threshold filtering

### Data Requirements

Your dataset should contain these columns:
- `college`: College/institution name
- `location`: Geographic location
- `university`: Affiliated university
- `course_offered`: Available courses/programs
- `ownership_type`: Institution type (Public/Private/Community)
- `phone_number`: Contact information (optional)
- `email`: Email contact (optional)

## 🔧 Configuration

### Customizing Weights

Modify weights in `recommender.py`:

```python
weights = {
    'course': 0.5,      # Course similarity weight
    'location': 0.2,    # Location similarity weight  
    'university': 0.2,  # University similarity weight
    'ownership': 0.1    # Ownership similarity weight
}
```

### Adding Location Alternatives

Extend geographic matching in `recommender.py`:

```python
self.location_alternatives = {
    'kathmandu': ['kathmandu valley', 'central region', 'bagmati'],
    'your_city': ['alternative1', 'alternative2']
}
```

## 📋 Example Output

```
🎯 YOUR PERSONALIZED COLLEGE RECOMMENDATIONS

🏆 #1 - Kathmandu University School of Engineering
   📍 Location: Dhulikhel, Kavre
   🏛️ University: Kathmandu University
   🎓 Course: Computer Engineering, Software Engineering, Civil Engineering
   🏢 Ownership: Private Institution
   ⭐ Match Score: 0.87

🏆 #2 - Tribhuvan University Institute of Engineering
   📍 Location: Pulchowk, Lalitpur  
   🏛️ University: Tribhuvan University
   🎓 Course: Computer Engineering, Electronics Engineering
   🏢 Ownership: Public Institution
   ⭐ Match Score: 0.82
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Nepal College Database
- **Libraries**: scikit-learn, pandas, streamlit, matplotlib
- **Evaluation Framework**: Custom precision and ranking metrics

#

---

**⭐ If this project helped you, please give it a star on GitHub!**

*Built with ❤️ for students seeking their perfect college match*
