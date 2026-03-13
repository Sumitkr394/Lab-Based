import csv
import pandas as pd

print("Processing local raw_news.csv...")
file_path = "raw_news.csv"

# We will read at most 3400 valid rows to avoid the EOF error at 3453
max_rows = 3400
valid_rows = []

try:
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Find index for title, text, label
        # In fake_or_real_news.csv, columns are usually: [Unnamed: 0, title, text, label]
        title_idx = header.index('title') if 'title' in header else -1
        text_idx = header.index('text') if 'text' in header else -1
        label_idx = header.index('label') if 'label' in header else -1
        
        count = 0
        for row in reader:
            if count >= max_rows:
                break
            try:
                # Need text and label at least
                if text_idx != -1 and label_idx != -1:
                    raw_text = row[text_idx]
                    raw_label = row[label_idx]
                    
                    if title_idx != -1:
                        raw_text = row[title_idx] + " " + raw_text
                        
                    raw_label = str(raw_label).capitalize() # FAKE -> Fake, REAL -> Real
                    if raw_label in ['Fake', 'Real']:
                        valid_rows.append({'text': raw_text, 'label': raw_label})
                        count += 1
            except Exception:
                continue
                
    # Now create dataframe and save safely
    df = pd.DataFrame(valid_rows)
    print(f"Dataset extracted. Shape: {df.shape}")
    df.to_csv("news.csv", index=False)
    print("Successfully saved real dataset to news.csv.")

except Exception as e:
    print(f"Error processing dataset: {e}")
