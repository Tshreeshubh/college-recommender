#!/usr/bin/env python3
"""
College Recommendation System - Streamlit GUI
Interactive web interface for getting personalized college recommendations
"""
import streamlit as st
import pandas as pd
import sys
import os
from typing import Optional, Tuple
import time

# Add the current directory to path to import recommender
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from recommender import recommend_colleges
except ImportError as e:
    st.error(f"❌ Error importing recommender module: {e}")
    st.error("Please ensure recommender.py is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="College Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load dataset to get dynamic options
@st.cache_data
def load_dataset():
    """Load the dataset and cache it for performance."""
    try:
        df = pd.read_csv("/Users/shreeshubh/suman sir assignment/data/cleaned_college_data.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found. Please check the file path.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()

# Load the data
df = load_dataset()

# Extract unique options from dataset
@st.cache_data
def get_unique_options():
    """Extract unique values for dropdowns."""
    if df.empty:
        return [], [], [], []
    
    # Get unique values and sort them
    universities = sorted([u for u in df['university'].dropna().unique() if str(u).strip()])
    locations = sorted([l for l in df['location'].dropna().unique() if str(l).strip()])
    ownership_types = sorted([o for o in df['ownership_type'].dropna().unique() if str(o).strip()])
    
    # Extract unique courses from course_offered column
    all_courses = []
    if 'course_offered' in df.columns:
        for courses in df['course_offered'].dropna():
            # Split by common delimiters and clean
            course_list = str(courses).replace(',', '|').replace(';', '|').replace('/', '|').split('|')
            for course in course_list:
                clean_course = course.strip()
                if clean_course and len(clean_course) > 2:  # Filter out very short entries
                    all_courses.append(clean_course)
    
    # Get unique courses and sort
    unique_courses = sorted(list(set(all_courses)))
    
    return universities, locations, ownership_types, unique_courses

# Get options from dataset
universities, locations, ownership_types, courses = get_unique_options()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .college-name {
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .college-details {
        font-size: 1rem;
        opacity: 0.9;
    }
    .match-score {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .help-text {
        background: #e3f2fd;
        padding: 1rem;
        border-left: 4px solid #2196f3;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stats-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def display_header():
    """Display the main header and description."""
    st.markdown('<h1 class="main-header">🎓 College Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Find the perfect college match for your academic journey!</p>', unsafe_allow_html=True)
    
    # Add some helpful information
    with st.expander("ℹ️ How to use this system", expanded=False):
        st.markdown("""
        **📖 Instructions:**
        - Select or type your preferences in the form below
        - You can leave fields blank if you don't have a specific preference
        - Use dropdowns for exact matches or text input for flexible search
        - The more specific you are, the better the recommendations
        
        **💡 Tips:**
        - Try different combinations if you don't get enough results
        - Use general terms for broader matches
        - Mix dropdown selections with text input for best results
        """)

def create_sidebar():
    """Create sidebar with additional options and information."""
    with st.sidebar:
        st.header("🔧 System Info & Stats")
        
        # Display dataset statistics
        if not df.empty:
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Colleges", len(df))
                st.metric("Universities", len(universities))
            with col2:
                st.metric("Locations", len(locations))
                st.metric("Unique Courses", len(courses))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show available options
        st.subheader("📋 Available Options")
        
        # Top universities
        if universities:
            with st.expander("🏛️ Top Universities"):
                for uni in universities[:10]:  # Show top 10
                    st.write(f"• {uni}")
                if len(universities) > 10:
                    st.write(f"... and {len(universities)-10} more")
        
        # Top locations
        if locations:
            with st.expander("📍 Available Locations"):
                for loc in locations[:10]:  # Show top 10
                    st.write(f"• {loc}")
                if len(locations) > 10:
                    st.write(f"... and {len(locations)-10} more")
        
        # Popular courses
        if courses:
            with st.expander("🎓 Popular Courses"):
                for course in courses[:15]:  # Show top 15
                    st.write(f"• {course}")
                if len(courses) > 15:
                    st.write(f"... and {len(courses)-15} more")

def display_recommendation_card(idx: int, row: pd.Series):
    """Display a single recommendation in a card format."""
    with st.container():
        st.markdown(f"""
        <div class="recommendation-card">
            <div class="college-name">🏆 #{idx} - {row.get('college', 'N/A')}</div>
            <div class="college-details">
                📍 <strong>Location:</strong> {row.get('location', 'N/A')}<br>
                🏛️ <strong>University:</strong> {row.get('university', 'N/A')}<br>
                🏢 <strong>Ownership:</strong> {row.get('ownership_type', 'N/A')}<br>
                🎓 <strong>Courses:</strong> {str(row.get('course_offered', 'N/A'))[:200]}{'...' if len(str(row.get('course_offered', ''))) > 200 else ''}
            </div>
            {f'<div class="match-score">⭐ Match Score: {row["score"]:.2f}</div>' if 'score' in row and pd.notna(row['score']) else ''}
        </div>
        """, unsafe_allow_html=True)

def validate_inputs(course: str, location: str, university: str) -> Tuple[bool, str]:
    """Validate user inputs."""
    if not any([course.strip(), location.strip(), university.strip()]):
        return False, "Please fill at least one field to get recommendations."
    return True, ""

def main():
    """Main application function."""
    # Check if dataset is loaded
    if df.empty:
        st.error("❌ Cannot load dataset. Please check the file path and try again.")
        return
    
    # Display header
    display_header()
    
    # Create sidebar
    create_sidebar()
    
    # Main form
    st.header("📝 Enter Your Preferences")
    
    with st.form("user_input_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎓 Course Preferences")
            
            # Course selection with both dropdown and text input
            course_method = st.radio(
                "Choose input method for course:",
                ["Select from dropdown", "Type custom course"],
                horizontal=True
            )
            
            if course_method == "Select from dropdown":
                user_course = st.selectbox(
                    "Select Course",
                    options=[""] + courses,
                    help="Choose from available courses in our database"
                )
            else:
                user_course = st.text_input(
                    "Enter Course Name",
                    placeholder="e.g., Computer Science, BBA, Civil Engineering",
                    help="Type the course you're interested in"
                )
            
            st.subheader("📍 Location Preferences")
            
            # Location selection
            location_method = st.radio(
                "Choose input method for location:",
                ["Select from dropdown", "Type custom location"],
                horizontal=True
            )
            
            if location_method == "Select from dropdown":
                user_location = st.selectbox(
                    "Select Location",
                    options=[""] + locations,
                    help="Choose from available locations"
                )
            else:
                user_location = st.text_input(
                    "Enter Location",
                    placeholder="e.g., Kathmandu, Pokhara, Chitwan",
                    help="Type your preferred location"
                )
        
        with col2:
            st.subheader("🏛️ University Preferences")
            
            # University selection
            university_method = st.radio(
                "Choose input method for university:",
                ["Select from dropdown", "Type custom university"],
                horizontal=True
            )
            
            if university_method == "Select from dropdown":
                user_university = st.selectbox(
                    "Select University",
                    options=[""] + universities,
                    help="Choose from available universities"
                )
            else:
                user_university = st.text_input(
                    "Enter University Name",
                    placeholder="e.g., Tribhuvan University, Kathmandu University",
                    help="Type your preferred university"
                )
            
            st.subheader("🏢 Institution Type")
            
            # Ownership type selection
            user_ownership = st.selectbox(
                "Select Institution Type",
                options=[""] + ownership_types,
                help="Select the type of institution you prefer"
            )
        
        # Number of recommendations
        st.subheader("📊 Recommendation Settings")
        top_n = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=20,
            value=5,
            help="Choose how many college recommendations you want to see"
        )
        
        # Submit button
        col_submit1, col_submit2, col_submit3 = st.columns([1, 2, 1])
        with col_submit2:
            submitted = st.form_submit_button(
                "🔍 Find My Perfect Colleges!",
                use_container_width=True,
                type="primary"
            )
    
    # Handle form submission
    if submitted:
        # Validate inputs
        is_valid, error_message = validate_inputs(user_course, user_location, user_university)
        if not is_valid:
            st.warning(f"❗ {error_message}")
            return
        
        # Show loading message
        with st.spinner("🔍 Searching for the best college matches for you..."):
            time.sleep(1)  # Small delay for better UX
            
            try:
                # Get recommendations
                recommendations = recommend_colleges(
                    user_course=user_course.strip() if user_course else "",
                    user_location=user_location.strip() if user_location else "",
                    user_university=user_university.strip() if user_university else "",
                    user_ownership=user_ownership if user_ownership else "",
                    top_n=top_n
                )
                
                # Display results
                if recommendations.empty:
                    st.error("❌ No recommendations found.")
                    
                    # Provide helpful suggestions
                    st.markdown("""
                    <div class="help-text">
                        <strong>💡 Try these suggestions:</strong><br>
                        • Use more general terms (e.g., "Engineering" instead of "Mechanical Engineering")<br>
                        • Leave some fields blank for broader search<br>
                        • Try different combinations of preferences<br>
                        • Check if your inputs match the available options in the sidebar
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ Found {len(recommendations)} perfect matches for you!")
                    
                    # Display recommendations in cards
                    st.header("🎯 Your Personalized Recommendations")
                    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                        display_recommendation_card(idx, row)
                    
                    # Option to download results
                    st.header("💾 Download Results")
                    csv = recommendations.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Recommendations as CSV",
                        data=csv,
                        file_name=f"college_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Show raw data option
                    with st.expander("📋 View Detailed Data Table", expanded=False):
                        st.dataframe(recommendations, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ An error occurred while getting recommendations: {str(e)}")
                st.error("Please check if all required files are present and try again.")
                
                # Show debug info in expander
                with st.expander("🔧 Debug Information", expanded=False):
                    st.code(f"Error details: {str(e)}")
                    st.code(f"Current working directory: {os.getcwd()}")
                    st.code(f"Python path: {sys.path}")

if __name__ == "__main__":
    main()
