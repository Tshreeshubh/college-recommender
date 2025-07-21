# enhanced_evaluate_with_topn.py

from recommender import recommend_colleges
import numpy as np
import pandas as pd
import re
from difflib import SequenceMatcher
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Expanded test cases with more diverse scenarios
test_cases = [
    {
        "course": "Computer Science",
        "location": "Kathmandu", 
        "university": "Tribhuvan University",
        "ownership": "private Institution",
        "expected_keywords": ["computer", "science", "it", "software", "programming", "technology", "cs", "computing", "informatics"],
        "expected_location": "Kathmandu",
        "expected_ownership": "private Institution",
        "alternative_locations": ["Kathmandu Valley", "Central Region"],
        "course_synonyms": ["BSc Computer Science", "Bachelor in Computer Science", "Computer Engineering"]
    },
    {
        "course": "Information Technology",
        "location": "Lalitpur",
        "university": "Tribhuvan University", 
        "ownership": "community Institution",
        "expected_keywords": ["information", "technology", "it", "computer", "software", "networking", "system", "data"],
        "expected_location": "Lalitpur",
        "expected_ownership": "community Institution",
        "alternative_locations": ["Kathmandu Valley", "Patan"],
        "course_synonyms": ["BSc IT", "Bachelor in Information Technology", "IT"]
    },
    {
        "course": "BBA",
        "location": "Bhaktapur",
        "university": "Westcliff University, CA, USA",
        "ownership": "private Institution",
        "expected_keywords": ["bba", "business", "administration", "management", "commerce", "bachelor", "business administration"],
        "expected_location": "Bhaktapur", 
        "expected_ownership": "private Institution",
        "alternative_locations": ["Kathmandu Valley"],
        "course_synonyms": ["Bachelor of Business Administration", "Business Administration", "Management"]
    },
    {
        "course": "Engineering",
        "location": "Pokhara",
        "university": "Pokhara University",
        "ownership": "public Institution",
        "expected_keywords": ["engineering", "civil", "mechanical", "electrical", "electronics", "computer", "technology", "technical"],
        "expected_location": "Pokhara",
        "expected_ownership": "public Institution",
        "alternative_locations": ["Gandaki Province", "Western Region"],
        "course_synonyms": ["Bachelor of Engineering", "BE", "B.Tech", "Engineering Technology"]
    },
    {
        "course": "Medicine",
        "location": "Chitwan",
        "university": "Tribhuvan University",
        "ownership": "public Institution",
        "expected_keywords": ["medicine", "medical", "mbbs", "doctor", "health", "clinical", "healthcare", "physician"],
        "expected_location": "Chitwan",
        "expected_ownership": "public Institution",
        "alternative_locations": ["Bharatpur", "Central Region"],
        "course_synonyms": ["MBBS", "Bachelor of Medicine", "Medical Science", "Clinical Medicine"]
    },
    {
        "course": "Nursing",
        "location": "Biratnagar",
        "university": "Purbanchal University",
        "ownership": "private Institution",
        "expected_keywords": ["nursing", "healthcare", "medical", "patient", "care", "health", "clinical", "nurse"],
        "expected_location": "Biratnagar",
        "expected_ownership": "private Institution",
        "alternative_locations": ["Eastern Region", "Morang"],
        "course_synonyms": ["BSc Nursing", "Bachelor of Nursing", "Nursing Science"]
    },
    {
        "course": "Agriculture",
        "location": "Janakpur",
        "university": "Agriculture and Forestry University",
        "ownership": "public Institution",
        "expected_keywords": ["agriculture", "farming", "crop", "soil", "agricultural", "agronomy", "horticulture", "livestock"],
        "expected_location": "Janakpur",
        "expected_ownership": "public Institution",
        "alternative_locations": ["Dhanusa", "Central Region"],
        "course_synonyms": ["BSc Agriculture", "Agricultural Science", "Agronomy"]
    },
    {
        "course": "MBA",
        "location": "Kathmandu",
        "university": "Kathmandu University",
        "ownership": "private Institution",
        "expected_keywords": ["mba", "master", "business", "administration", "management", "executive", "leadership"],
        "expected_location": "Kathmandu",
        "expected_ownership": "private Institution",
        "alternative_locations": ["Kathmandu Valley", "Central Region"],
        "course_synonyms": ["Master of Business Administration", "Masters in Business", "Executive MBA"]
    }
]

class TopNAccuracyEvaluator:
    def __init__(self):
        self.total_precision_scores = []
        self.total_content_relevance = []
        self.total_location_accuracy = []
        self.total_ownership_accuracy = []
        self.top_n_results = defaultdict(list)  # Store results for different N values
        self.debug_mode = False  # Simplified for cleaner output
        self.n_values = [1, 3, 5, 7, 10, 15, 20]  # Different N values to test
        
    def fuzzy_string_match(self, str1, str2, threshold=0.6):
        """Enhanced fuzzy string matching"""
        if pd.isna(str1) or pd.isna(str2):
            return 0.0
        return SequenceMatcher(None, str(str1).lower(), str(str2).lower()).ratio()
    
    def enhanced_content_relevance(self, course_offered, test_case):
        """Enhanced content relevance with multiple matching strategies"""
        if pd.isna(course_offered):
            return 0.0
            
        course_lower = str(course_offered).lower()
        
        # 1. Exact keyword matching
        keyword_matches = sum(1 for keyword in test_case['expected_keywords'] if keyword in course_lower)
        keyword_score = keyword_matches / len(test_case['expected_keywords'])
        
        # 2. Synonym matching with fuzzy logic
        synonym_score = 0.0
        if 'course_synonyms' in test_case:
            synonym_matches = [self.fuzzy_string_match(course_offered, synonym) 
                             for synonym in test_case['course_synonyms']]
            synonym_score = max(synonym_matches) if synonym_matches else 0.0
        
        # 3. Fuzzy matching with original course
        fuzzy_score = self.fuzzy_string_match(course_offered, test_case['course'])
        
        # 4. Partial word matching
        course_words = set(course_lower.split())
        expected_words = set(test_case['course'].lower().split())
        word_overlap = len(course_words.intersection(expected_words)) / max(len(expected_words), 1)
        
        # Combined score with adjusted weights
        final_score = (0.3 * keyword_score + 0.25 * synonym_score + 
                      0.25 * fuzzy_score + 0.2 * word_overlap)
        
        return final_score
    
    def enhanced_location_accuracy(self, recommended_location, test_case):
        """Enhanced location matching with geographic awareness"""
        if pd.isna(recommended_location):
            return 0.0
            
        location_str = str(recommended_location).strip().lower()
        expected_location = str(test_case['expected_location']).strip().lower()
        
        # 1. Exact match (highest score)
        if location_str == expected_location:
            return 1.0
        
        # 2. Fuzzy match with main location
        fuzzy_main = self.fuzzy_string_match(location_str, expected_location)
        
        # 3. Alternative location matching
        fuzzy_alt = 0.0
        if 'alternative_locations' in test_case:
            alt_scores = [self.fuzzy_string_match(location_str, alt.lower()) 
                         for alt in test_case['alternative_locations']]
            fuzzy_alt = max(alt_scores) if alt_scores else 0.0
        
        # 4. Substring matching for compound locations
        substring_score = 0.0
        if expected_location in location_str or location_str in expected_location:
            substring_score = 0.7
        
        # Take the best match with appropriate weighting
        final_score = max(fuzzy_main, fuzzy_alt * 0.8, substring_score)
        
        return final_score
    
    def enhanced_ownership_accuracy(self, recommended_ownership, expected_ownership):
        """Enhanced ownership matching with normalization"""
        if pd.isna(recommended_ownership):
            return 0.0
            
        # Normalize ownership strings
        rec_clean = re.sub(r'[^\w\s]', '', str(recommended_ownership)).strip().lower()
        exp_clean = re.sub(r'[^\w\s]', '', str(expected_ownership)).strip().lower()
        
        # 1. Exact match after normalization
        if rec_clean == exp_clean:
            return 1.0
        
        # 2. Keyword-based matching for ownership types
        ownership_mapping = {
            'private': ['private', 'pvt', 'proprietary'],
            'public': ['public', 'government', 'govt', 'state'],
            'community': ['community', 'local', 'municipal'],
            'autonomous': ['autonomous', 'independent'],
            'international': ['international', 'foreign']
        }
        
        keyword_score = 0.0
        for ownership_type, keywords in ownership_mapping.items():
            exp_match = any(kw in exp_clean for kw in keywords)
            rec_match = any(kw in rec_clean for kw in keywords)
            if exp_match and rec_match:
                keyword_score = 1.0
                break
        
        # 3. Fuzzy matching as fallback
        fuzzy_score = self.fuzzy_string_match(rec_clean, exp_clean)
        
        final_score = max(keyword_score, fuzzy_score)
        
        return final_score
    
    def calculate_overall_relevance(self, row, test_case):
        """Calculate comprehensive relevance score"""
        content_score = self.enhanced_content_relevance(row['course_offered'], test_case)
        location_score = self.enhanced_location_accuracy(row['location'], test_case)
        ownership_score = self.enhanced_ownership_accuracy(row['ownership_type'], test_case['expected_ownership'])
        
        # Weighted combination with emphasis on content
        overall_score = (0.6 * content_score + 0.25 * location_score + 0.15 * ownership_score)
        
        return {
            'content_relevance': content_score,
            'location_accuracy': location_score,
            'ownership_accuracy': ownership_score,
            'overall_relevance': overall_score
        }
    
    def precision_at_k(self, results, test_case, k, threshold=0.4):
        """Calculate Precision@K - Simple and clean implementation"""
        if results.empty or k == 0:
            return 0.0
            
        top_k = results.head(k)
        relevant_count = 0
        
        for idx, row in top_k.iterrows():
            relevance = self.calculate_overall_relevance(row, test_case)
            if relevance['overall_relevance'] >= threshold:
                relevant_count += 1
        
        return relevant_count / k
    
    def precision_at_n(self, results, test_case, n, threshold=0.4):
        """Calculate precision at N with adaptive threshold"""
        if results.empty or n == 0:
            return 0.0, []
            
        top_n = results.head(n)
        relevance_scores = []
        detailed_results = []
        
        for idx, row in top_n.iterrows():
            relevance = self.calculate_overall_relevance(row, test_case)
            relevance_scores.append(relevance['overall_relevance'])
            detailed_results.append({
                'rank': idx + 1,
                'college': row['college'],
                'relevance_score': relevance['overall_relevance'],
                'is_relevant': relevance['overall_relevance'] >= threshold,
                'content_score': relevance['content_relevance'],
                'location_score': relevance['location_accuracy'],
                'ownership_score': relevance['ownership_accuracy']
            })
        
        relevant_count = sum(1 for score in relevance_scores if score >= threshold)
        precision = relevant_count / n
        
        return precision, detailed_results
    
    def top_n_accuracy(self, results, test_case, n, threshold=0.4):
        """Calculate top-N accuracy (whether at least one relevant item in top N)"""
        if results.empty or n == 0:
            return 0.0
            
        top_n = results.head(n)
        
        for idx, row in top_n.iterrows():
            relevance = self.calculate_overall_relevance(row, test_case)
            if relevance['overall_relevance'] >= threshold:
                return 1.0  # At least one relevant item found
        
        return 0.0  # No relevant items found
    
    def mean_reciprocal_rank(self, results, test_case, threshold=0.4):
        """Calculate Mean Reciprocal Rank"""
        if results.empty:
            return 0.0
            
        for idx, row in results.iterrows():
            relevance = self.calculate_overall_relevance(row, test_case)
            if relevance['overall_relevance'] >= threshold:
                return 1.0 / (idx + 1)  # Return reciprocal of rank (1-indexed)
        
        return 0.0  # No relevant items found
    
    def ndcg_at_n(self, results, test_case, n):
        """Calculate Normalized Discounted Cumulative Gain at N"""
        if results.empty or n == 0:
            return 0.0
        
        top_n = results.head(n)
        dcg = 0.0
        
        # Calculate DCG
        for idx, row in top_n.iterrows():
            relevance = self.calculate_overall_relevance(row, test_case)
            rel_score = relevance['overall_relevance']
            # DCG formula: rel_score / log2(rank + 1)
            dcg += rel_score / np.log2(idx + 2)  # idx is 0-based, so +2
        
        # Calculate IDCG (ideal DCG) - assume perfect ordering
        relevance_scores = []
        for idx, row in results.iterrows():
            relevance = self.calculate_overall_relevance(row, test_case)
            relevance_scores.append(relevance['overall_relevance'])
        
        # Sort in descending order for ideal ranking
        ideal_scores = sorted(relevance_scores, reverse=True)[:n]
        idcg = sum(score / np.log2(idx + 2) for idx, score in enumerate(ideal_scores))
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_single_case(self, case_num, test_case):
        """Enhanced single case evaluation with simple Precision@K output"""
        # Get recommendations
        try:
            results = recommend_colleges(
                user_course=test_case['course'],
                user_location=test_case['location'],
                user_university=test_case['university'],
                user_ownership=test_case['ownership'],
                top_n=20  # Get more results for comprehensive evaluation
            )
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return None
        
        if results.empty:
            print("No recommendations found!")
            return None
        
        # Calculate Precision@K for different K values
        precision_k_results = {}
        for k in [1, 3, 5, 10]:
            precision_k_results[f'precision_at_{k}'] = self.precision_at_k(results, test_case, k)
        
        # Simple output format as requested
        print(f"Test Case [{test_case['course']}, {test_case['location']}, {test_case['university']}] → Precision@5: {precision_k_results['precision_at_5']:.2f}")
        
        # Calculate other metrics for comprehensive analysis
        metrics = {}
        
        # Precision@N for different N values
        precision_scores = {}
        top_n_accuracies = {}
        ndcg_scores = {}
        
        for n in self.n_values:
            precision, detailed = self.precision_at_n(results, test_case, n)
            top_n_acc = self.top_n_accuracy(results, test_case, n)
            ndcg = self.ndcg_at_n(results, test_case, n)
            
            precision_scores[f'precision_at_{n}'] = precision
            top_n_accuracies[f'top_{n}_accuracy'] = top_n_acc
            ndcg_scores[f'ndcg_at_{n}'] = ndcg
        
        # Mean Reciprocal Rank
        mrr = self.mean_reciprocal_rank(results, test_case)
        
        # Store results
        metrics.update(precision_scores)
        metrics.update(top_n_accuracies)
        metrics.update(ndcg_scores)
        metrics.update(precision_k_results)  # Add Precision@K results
        metrics['mrr'] = mrr
        
        return metrics
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation with simple Precision@K output"""
        print("🚀 COMPREHENSIVE EVALUATION WITH PRECISION@K")
        print("=" * 60)
        
        all_results = []
        
        for i, case in enumerate(test_cases):
            result = self.evaluate_single_case(i+1, case)
            if result:
                all_results.append(result)
        
        if all_results:
            self.print_simple_summary(all_results)
        else:
            print("❌ No successful evaluations completed!")
        
        return all_results
    
    def print_simple_summary(self, all_results):
        """Print simple summary with Precision@K focus"""
        print("\n" + "=" * 60)
        print("📊 PRECISION@K SUMMARY")
        print("=" * 60)
        
        # Calculate average Precision@K metrics
        k_values = [1, 3, 5, 10]
        print("Average Precision@K Scores:")
        for k in k_values:
            key = f'precision_at_{k}'
            avg_score = np.mean([result[key] for result in all_results])
            print(f"  Precision@{k}: {avg_score:.3f} ({avg_score*100:.1f}%)")
        
        # Overall performance
        avg_p5 = np.mean([result['precision_at_5'] for result in all_results])
        print(f"\n🎯 Overall System Performance (Precision@5): {avg_p5:.3f}")
        
        if avg_p5 >= 0.8:
            print("Performance Grade: EXCELLENT ⭐⭐⭐⭐⭐")
        elif avg_p5 >= 0.6:
            print("Performance Grade: GOOD ⭐⭐⭐⭐")
        elif avg_p5 >= 0.4:
            print("Performance Grade: FAIR ⭐⭐⭐")
        else:
            print("Performance Grade: NEEDS IMPROVEMENT ⭐")

def run_comprehensive_top_n_evaluation():
    """Main function to run the comprehensive evaluation with Precision@K"""
    evaluator = TopNAccuracyEvaluator()
    results = evaluator.run_comprehensive_evaluation()
    return results

if __name__ == "__main__":
    run_comprehensive_top_n_evaluation()
