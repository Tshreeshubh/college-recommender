import pandas as pd
import numpy as np

def aggressive_duplicate_removal():
    # Load the dataset
    file_path = "/Users/shreeshubh/suman sir assignment/data/cleaned_college_data.csv"
    
    try:
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    

    print(f"\nFirst 5 rows of original data:")
    print(df.head())
    

    print(f"\n{'='*60}")
    print("DUPLICATE ANALYSIS")
    print(f"{'='*60}")
    
   
    exact_dups = df.duplicated().sum()
    print(f"Exact duplicates (all columns identical): {exact_dups}")
    

    df_reset = df.reset_index(drop=True)
    exact_dups_reset = df_reset.duplicated().sum()
    print(f"Exact duplicates after index reset: {exact_dups_reset}")
    
    # Start with a copy
    df_clean = df.copy()
    original_count = len(df_clean)
    
    # Remove exact duplicates first
    df_clean = df_clean.drop_duplicates()
    after_exact = len(df_clean)
    print(f"After removing exact duplicates: {after_exact} rows (removed {original_count - after_exact})")
    

    text_cols = []
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            text_cols.append(col)
          
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = df_clean[col].str.strip()
            df_clean[col] = df_clean[col].str.lower()
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
            df_clean[col] = df_clean[col].str.replace(r'[^\w\s]', '', regex=True)
            df_clean[col] = df_clean[col].replace('nan', np.nan)
    
    print(f"Cleaned text in columns: {text_cols}")
    
    # Remove duplicates after text cleaning
    before_text_clean = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after_text_clean = len(df_clean)
    print(f"After text cleaning and duplicate removal: {after_text_clean} rows (removed {before_text_clean - after_text_clean})")
    
    # Check for specific column duplicates
    if 'university' in df_clean.columns:
        # University-based duplicates
        uni_dups = df_clean.duplicated(subset=['university']).sum()
        print(f"University name duplicates: {uni_dups}")
        
        if uni_dups > 0:
            print("Sample university duplicates:")
            uni_dup_mask = df_clean.duplicated(subset=['university'], keep=False)
            sample_uni_dups = df_clean[uni_dup_mask]['university'].value_counts().head()
            print(sample_uni_dups)
    

    key_columns = []
    potential_keys = ['university', 'college', 'name', 'institution']
    
    for col in df_clean.columns:
        col_lower = col.lower()
        if any(key in col_lower for key in potential_keys):
            key_columns.append(col)
    
    if key_columns:
        print(f"\nUsing key columns for duplicate removal: {key_columns}")
        before_key = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=key_columns, keep='first')
        after_key = len(df_clean)
        print(f"After key-based duplicate removal: {after_key} rows (removed {before_key - after_key})")
    
   
    print(f"\nChecking for potential near-duplicates...")
    if len(df_clean) > 1:
        # Sample check for similar rows
        sample_size = min(10, len(df_clean))
        print(f"Sample of remaining data (first {sample_size} rows):")
        print(df_clean.head(sample_size).to_string())
    

    df_clean = df_clean.reset_index(drop=True)
    

    df_clean = df_clean.dropna(how='all')
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Original rows: {original_count}")
    print(f"Final rows: {len(df_clean)}")
    print(f"Total rows removed: {original_count - len(df_clean)}")
    print(f"Percentage removed: {((original_count - len(df_clean)) / original_count * 100):.1f}%")
    

    output_path = "/Users/shreeshubh/suman sir assignment/data/cleaned_college_data_no_duplicates.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned dataset saved to: {output_path}")
    

    if key_columns:
        for col in key_columns[:2]:  
            print(f"\nValue counts for {col} (showing duplicates if any):")
            counts = df_clean[col].value_counts()
            duplicates = counts[counts > 1]
            if len(duplicates) > 0:
                print("⚠️ Still has duplicates:")
                print(duplicates.head())
            else:
                print("✅ No duplicates found")
                print(f"Unique values: {len(counts)}")
    
    return df_clean

# Run the aggressive cleaning
if __name__ == "__main__":
    cleaned_data = aggressive_duplicate_removal()