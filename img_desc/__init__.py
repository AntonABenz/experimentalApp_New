import pandas as pd
import json
import ast

# 1. Load Data
# -------------------------------------------------------
file_path = 'data.csv'  # Make sure this matches your downloaded filename
df = pd.read_csv(file_path)

print(f"Loaded {len(df)} rows.")

# 2. Clean Sentences (Parse JSON)
# -------------------------------------------------------
def clean_sentences(row):
    """
    Converts '[["Some", "the A"], ["None", "the B"]]' 
    into 'Some the A; None the B'
    """
    raw = row.get('sentences', '[]')
    try:
        # If it's a string, parse it as JSON
        if isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = raw
        
        if not isinstance(data, list): return ""
        
        # Combine ["Quantifier", "Object"] pairs into strings
        readable_list = [f"{item[0]} {item[1]}" for item in data if len(item) >= 2]
        return "; ".join(readable_list)
    except:
        return ""

# Apply cleaning
df['clean_sentences'] = df.apply(clean_sentences, axis=1)

# 3. Analyze Time Taken (Seconds)
# -------------------------------------------------------
# Convert seconds to numeric, forcing errors to NaN
df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')

print("\n--- TIMING ANALYSIS (Seconds per Round) ---")
print(df.groupby('role')['seconds'].describe())

# 4. Save Cleaned File
# -------------------------------------------------------
output_cols = [
    'participant', 'round', 'role', 'condition', 'item_nr', 
    'image', 'clean_sentences', 'rewards', 'seconds'
]

# Save only the useful columns
df[output_cols].to_csv('cleaned_results.csv', index=False)
print("\n✅ Success! Saved cleaned data to 'cleaned_results.csv'")
