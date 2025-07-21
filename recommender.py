# improved_recommender.py
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import re
from collections import defaultdict
import os

# Load data and models with error handling
def load_models():
    try:
        df = pd.read_csv("data/cleaned_college_data.csv")
        
        # Check if model files exist
        if not os.path.exists("models/course_matrix.pkl"):
            print("Warning: course_matrix.pkl not found. Creating new TF-IDF matrix...")
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Create new TF-IDF vectorizer
            tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
            course_matrix = tfidf.fit_transform(df['course_offered'].fillna(''))
            
            # Save the models
            os.makedirs("models", exist_ok=True)
            joblib.dump(course_matrix, "models/course_matrix.pkl")
            joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
        else:
            course_matrix = joblib.load("models/course_matrix.pkl")
            tfidf = joblib.load("models/tfidf_vectorizer.pkl")
            
            # Check shape compatibility
            if course_matrix.shape[0] != len(df):
                print(f"Warning: Shape mismatch detected!")
                print(f"Course matrix has {course_matrix.shape[0]} samples, but DataFrame has {len(df)} rows")
                print("Recreating TF-IDF matrix to match current dataset...")
                
                from sklearn.feature_extraction.text import TfidfVectorizer
                tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
                course_matrix = tfidf.fit_transform(df['course_offered'].fillna(''))
                
                # Save the updated models
                joblib.dump(course_matrix, "models/course_matrix.pkl")
                joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
        
        # Load or create encoders
        encoders = {}
        for col, encoder_name in [('location', 'location_encoder'), 
                                 ('university', 'university_encoder'), 
                                 ('ownership_type', 'ownership_encoder')]:
            encoder_path = f"models/{encoder_name}.pkl"
            if os.path.exists(encoder_path):
                encoders[col] = joblib.load(encoder_path)
            else:
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()
                encoder.fit(df[col].fillna('unknown'))
                encoders[col] = encoder
                joblib.dump(encoder, encoder_path)
        
        return df, course_matrix, tfidf, encoders
    
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

# Load data and models
df, course_matrix, tfidf, encoders = load_models()
le_location = encoders['location']
le_university = encoders['university']
le_ownership = encoders['ownership_type']

class EnhancedCollegeRecommender:
    def __init__(self):
        self.df = df
        self.course_matrix = course_matrix
        self.tfidf = tfidf
        self.le_location = le_location
        self.le_university = le_university
        self.le_ownership = le_ownership
        
        # Verify shape consistency
        print(f"DataFrame shape: {self.df.shape}")
        print(f"Course matrix shape: {self.course_matrix.shape}")
        
        if self.course_matrix.shape[0] != len(self.df):
            raise ValueError(f"Shape mismatch: course_matrix has {self.course_matrix.shape[0]} rows, but DataFrame has {len(self.df)} rows")
        
        # Create lookup dictionaries for fuzzy matching
        self.location_lookup = self._create_fuzzy_lookup(df['location'].unique())
        self.university_lookup = self._create_fuzzy_lookup(df['university'].unique())
        self.ownership_lookup = self._create_fuzzy_lookup(df['ownership_type'].unique())
        
        # Location hierarchy and alternatives
        self.location_alternatives = {
            'kathmandu': ['kathmandu valley', 'central region', 'bagmati'],
            'lalitpur': ['kathmandu valley', 'central region', 'patan', 'bagmati'],
            'bhaktapur': ['kathmandu valley', 'central region', 'bagmati'],
            'pokhara': ['western region', 'gandaki', 'kaski'],
            'chitwan': ['central region', 'bagmati'],
            'biratnagar': ['eastern region', 'koshi', 'morang']
        }
        
        # Ownership type synonyms
        self.ownership_synonyms = {
            'private': ['private institution', 'pvt', 'private college'],
            'public': ['public institution', 'government', 'govt', 'public college'],
            'community': ['community institution', 'local', 'community college']
        }
    
    def _create_fuzzy_lookup(self, values):
        """Create a dictionary for fuzzy string matching"""
        lookup = {}
        for value in values:
            if pd.notna(value):
                cleaned = str(value).lower().strip()
                lookup[cleaned] = value
        return lookup
    
    def _fuzzy_match(self, user_input, lookup_dict, threshold=0.6):
        """Find the best fuzzy match in lookup dictionary"""
        if not user_input or pd.isna(user_input):
            return None, 0.0
        
        user_clean = str(user_input).lower().strip()
        
        # Exact match first
        if user_clean in lookup_dict:
            return lookup_dict[user_clean], 1.0
        
        # Fuzzy matching
        best_match = None
        best_score = 0.0
        
        for key, value in lookup_dict.items():
            score = SequenceMatcher(None, user_clean, key).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = value
        
        return best_match, best_score
    
    def _enhanced_location_similarity(self, user_location, df_locations):
        """Enhanced location similarity with fuzzy matching and alternatives"""
        if not user_location or pd.isna(user_location):
            return np.zeros(len(df_locations))
        
        user_clean = str(user_location).lower().strip()
        similarities = np.zeros(len(df_locations))
        
        for i, df_location in enumerate(df_locations):
            if pd.isna(df_location):
                continue
                
            df_clean = str(df_location).lower().strip()
            
            # Exact match
            if user_clean == df_clean:
                similarities[i] = 1.0
                continue
            
            # Fuzzy match
            fuzzy_score = SequenceMatcher(None, user_clean, df_clean).ratio()
            
            # Alternative location matching
            alt_score = 0.0
            if user_clean in self.location_alternatives:
                for alt in self.location_alternatives[user_clean]:
                    alt_fuzzy = SequenceMatcher(None, alt, df_clean).ratio()
                    alt_score = max(alt_score, alt_fuzzy)
            
            # Reverse alternative matching
            for key, alternatives in self.location_alternatives.items():
                if df_clean == key and user_clean in alternatives:
                    alt_score = max(alt_score, 0.8)  # High score for reverse match
            
            # Take the best score
            similarities[i] = max(fuzzy_score, alt_score * 0.9)  # Slight penalty for alternatives
        
        return similarities
    
    def _enhanced_university_similarity(self, user_university, df_universities):
        """Enhanced university similarity with fuzzy matching"""
        if not user_university or pd.isna(user_university):
            return np.zeros(len(df_universities))
        
        user_clean = str(user_university).lower().strip()
        similarities = np.zeros(len(df_universities))
        
        for i, df_university in enumerate(df_universities):
            if pd.isna(df_university):
                continue
                
            df_clean = str(df_university).lower().strip()
            
            # Exact match
            if user_clean == df_clean:
                similarities[i] = 1.0
                continue
            
            # Fuzzy match
            similarities[i] = SequenceMatcher(None, user_clean, df_clean).ratio()
            
            # Keyword matching for universities
            user_words = set(user_clean.split())
            df_words = set(df_clean.split())
            
            # Jaccard similarity for word overlap
            if user_words and df_words:
                jaccard = len(user_words & df_words) / len(user_words | df_words)
                similarities[i] = max(similarities[i], jaccard)
        
        return similarities
    
    def _enhanced_ownership_similarity(self, user_ownership, df_ownerships):
        """Enhanced ownership similarity with synonyms and fuzzy matching"""
        if not user_ownership or pd.isna(user_ownership):
            return np.zeros(len(df_ownerships))
        
        user_clean = str(user_ownership).lower().strip()
        similarities = np.zeros(len(df_ownerships))
        
        # Normalize ownership type
        user_normalized = self._normalize_ownership(user_clean)
        
        for i, df_ownership in enumerate(df_ownerships):
            if pd.isna(df_ownership):
                continue
                
            df_clean = str(df_ownership).lower().strip()
            df_normalized = self._normalize_ownership(df_clean)
            
            # Exact match after normalization
            if user_normalized == df_normalized:
                similarities[i] = 1.0
                continue
            
            # Fuzzy match
            similarities[i] = SequenceMatcher(None, user_clean, df_clean).ratio()
        
        return similarities
    
    def _normalize_ownership(self, ownership_text):
        """Normalize ownership type to standard categories"""
        ownership_clean = re.sub(r'[^\w\s]', '', ownership_text).strip()
        
        for standard, synonyms in self.ownership_synonyms.items():
            if any(syn in ownership_clean for syn in synonyms):
                return standard
        
        return ownership_clean
    
    def _enhanced_course_similarity(self, user_course, course_matrix):
        """Enhanced course similarity with preprocessing"""
        if not user_course or pd.isna(user_course):
            return np.zeros(course_matrix.shape[0])
        
        # Preprocess user course
        processed_course = self._preprocess_course_text(user_course)
        
        try:
            # Vectorize with TF-IDF
            user_course_vec = self.tfidf.transform([processed_course])
            
            # Compute cosine similarity
            course_sim = cosine_similarity(user_course_vec, course_matrix).flatten()
            
            # Ensure the shape matches the DataFrame
            if course_sim.shape[0] != len(self.df):
                print(f"Warning: Course similarity shape mismatch. Truncating or padding to match DataFrame size.")
                if course_sim.shape[0] > len(self.df):
                    course_sim = course_sim[:len(self.df)]
                else:
                    # Pad with zeros if needed
                    padded_sim = np.zeros(len(self.df))
                    padded_sim[:course_sim.shape[0]] = course_sim
                    course_sim = padded_sim
            
            return course_sim
            
        except Exception as e:
            print(f"Error in course similarity calculation: {e}")
            return np.zeros(len(self.df))
    
    def _preprocess_course_text(self, course_text):
        """Preprocess course text for better matching"""
        if not course_text:
            return ""
        
        # Convert to lowercase
        processed = str(course_text).lower()
        
        # Expand common abbreviations
        abbreviations = {
            'cs': 'computer science',
            'it': 'information technology',
            'bba': 'bachelor of business administration',
            'mba': 'master of business administration',
            'bsc': 'bachelor of science',
            'msc': 'master of science',
            'be': 'bachelor of engineering',
            'me': 'master of engineering'
        }
        
        for abbr, full in abbreviations.items():
            processed = re.sub(r'\b' + abbr + r'\b', full, processed)
        
        return processed
    
    def recommend_colleges(self, user_course, user_location, user_university, user_ownership, top_n=5):
        """Enhanced recommendation function with improved similarity calculations"""
        
        # Enhanced course similarity
        course_sim = self._enhanced_course_similarity(user_course, self.course_matrix)
        
        # Enhanced categorical similarities
        location_sim = self._enhanced_location_similarity(user_location, self.df['location'])
        university_sim = self._enhanced_university_similarity(user_university, self.df['university'])
        ownership_sim = self._enhanced_ownership_similarity(user_ownership, self.df['ownership_type'])
        
        # Ensure all similarity arrays have the same length
        n_rows = len(self.df)
        course_sim = course_sim[:n_rows] if len(course_sim) > n_rows else course_sim
        location_sim = location_sim[:n_rows] if len(location_sim) > n_rows else location_sim
        university_sim = university_sim[:n_rows] if len(university_sim) > n_rows else university_sim
        ownership_sim = ownership_sim[:n_rows] if len(ownership_sim) > n_rows else ownership_sim
        
        # Pad with zeros if any array is shorter
        if len(course_sim) < n_rows:
            padded = np.zeros(n_rows)
            padded[:len(course_sim)] = course_sim
            course_sim = padded
        
        if len(location_sim) < n_rows:
            padded = np.zeros(n_rows)
            padded[:len(location_sim)] = location_sim
            location_sim = padded
            
        if len(university_sim) < n_rows:
            padded = np.zeros(n_rows)
            padded[:len(university_sim)] = university_sim
            university_sim = padded
            
        if len(ownership_sim) < n_rows:
            padded = np.zeros(n_rows)
            padded[:len(ownership_sim)] = ownership_sim
            ownership_sim = padded
        
        # Adaptive weighting based on input quality
        weights = self._calculate_adaptive_weights(
            user_course, user_location, user_university, user_ownership
        )
        
        # Combined similarity score
        final_score = (
            weights['course'] * course_sim +
            weights['location'] * location_sim +
            weights['university'] * university_sim +
            weights['ownership'] * ownership_sim
        )
        
        # Apply filters for minimum thresholds
        valid_indices = self._apply_filters(
            course_sim, location_sim, university_sim, ownership_sim
        )
        
        # Zero out invalid recommendations
        final_score = final_score * valid_indices
        
        # Get top N recommendations
        top_indices = final_score.argsort()[-top_n * 2:][::-1]  # Get more candidates
        
        # Filter out zero-score recommendations
        valid_top_indices = [i for i in top_indices if final_score[i] > 0.05][:top_n]
        
        if not valid_top_indices:
            # Fallback: return best matches even if scores are low
            valid_top_indices = top_indices[:top_n]
        
        # Create results DataFrame
        results = self.df.iloc[valid_top_indices][
            ['college', 'location', 'university', 'course_offered', 'ownership_type']
        ].copy()
        
        results['score'] = final_score[valid_top_indices].round(3)
        
        # Add relevance metrics for debugging
        results['course_similarity'] = course_sim[valid_top_indices].round(3)
        results['location_similarity'] = location_sim[valid_top_indices].round(3)
        results['university_similarity'] = university_sim[valid_top_indices].round(3)
        results['ownership_similarity'] = ownership_sim[valid_top_indices].round(3)
        
        return results.reset_index(drop=True)
    
    def _calculate_adaptive_weights(self, user_course, user_location, user_university, user_ownership):
        """Calculate adaptive weights based on input specificity"""
        weights = {
            'course': 0.5,      # Base weight for course
            'location': 0.25,   # Updated weight for location (was 0.2)
            'university': 0.15, # Updated weight for university (was 0.2)
            'ownership': 0.1    # Base weight for ownership
        }
        
        # Adjust weights based on input quality
        if not user_course or len(str(user_course).strip()) < 3:
            weights['course'] *= 0.5
        
        if not user_location or len(str(user_location).strip()) < 2:
            weights['location'] *= 0.5
        else:
            # Boost location weight if it's specific
            weights['location'] *= 1.2
        
        if not user_university or len(str(user_university).strip()) < 3:
            weights['university'] *= 0.5
        
        if not user_ownership or len(str(user_ownership).strip()) < 3:
            weights['ownership'] *= 0.5
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight
        
        return weights
    
    def _apply_filters(self, course_sim, location_sim, university_sim, ownership_sim):
        """Apply minimum threshold filters"""
        # Minimum thresholds
        min_course_threshold = 0.1
        min_location_threshold = 0.1
        min_overall_threshold = 0.05
        
        # Create filter mask
        course_filter = course_sim >= min_course_threshold
        location_filter = location_sim >= min_location_threshold
        overall_filter = (course_sim + location_sim + university_sim + ownership_sim) >= min_overall_threshold
        
        # Combine filters (at least course OR location must meet threshold)
        valid_mask = (course_filter | location_filter) & overall_filter
        
        return valid_mask.astype(float)

# Create global recommender instance
recommender = EnhancedCollegeRecommender()

def recommend_colleges(user_course, user_location, user_university, user_ownership, top_n=5):
    """
    Enhanced recommendation function (maintains compatibility with existing code)
    """
    return recommender.recommend_colleges(
        user_course, user_location, user_university, user_ownership, top_n
    )

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
    
    print("🎓 Enhanced College Recommendations:")
    print(recommendations)
    
    # Show detailed analysis
    print("\n📊 Detailed Similarity Analysis:")
    for idx, row in recommendations.iterrows():
        print(f"\n{idx+1}. {row['college']}")
        print(f"   Overall Score: {row['score']:.3f}")
        print(f"   Course Similarity: {row['course_similarity']:.3f}")
        print(f"   Location Similarity: {row['location_similarity']:.3f}")
        print(f"   University Similarity: {row['university_similarity']:.3f}")
        print(f"   Ownership Similarity: {row['ownership_similarity']:.3f}")