"""
Helper script to download and prepare the Fake and Real News Dataset from Kaggle.

This script helps you download the public dataset from Kaggle and prepare it for use.
"""

import pandas as pd
import os

def download_kaggle_dataset():
    """
    Instructions and helper function to download the Fake and Real News Dataset.
    
    The dataset is available at:
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    """
    print("="*60)
    print("Downloading Fake and Real News Dataset")
    print("="*60)
    print("\nThe dataset is available on Kaggle.")
    print("You have two options:\n")
    
    print("OPTION 1: Download manually from Kaggle")
    print("-" * 60)
    print("1. Visit: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    print("2. Click 'Download' (you may need to create a free Kaggle account)")
    print("3. Extract the zip file")
    print("4. You'll find 'True.csv' and 'Fake.csv' files")
    print("5. Place them in this directory")
    print("6. Run: python download_dataset.py")
    print()
    
    print("OPTION 2: Use Kaggle API (recommended)")
    print("-" * 60)
    print("1. Install Kaggle API: pip install kaggle")
    print("2. Set up Kaggle credentials (see: https://github.com/Kaggle/kaggle-api)")
    print("3. Run: kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset")
    print("4. Unzip the downloaded file")
    print("5. Run: python download_dataset.py")
    print()
    
    # Check if files already exist
    if os.path.exists('True.csv') and os.path.exists('Fake.csv'):
        print("Found True.csv and Fake.csv files!")
        print("Running prepare_dataset.py to combine them...")
        prepare_dataset()
    else:
        print("True.csv and Fake.csv not found in current directory.")
        print("Please download them first using one of the options above.")


def prepare_dataset():
    """
    Combine True.csv and Fake.csv into a single dataset.csv file.
    """
    if not os.path.exists('True.csv') or not os.path.exists('Fake.csv'):
        print("Error: True.csv and Fake.csv not found!")
        print("Please download them from Kaggle first.")
        return
    
    print("\nCombining True.csv and Fake.csv...")
    
    try:
        # Load the datasets
        true_news = pd.read_csv('True.csv')
        fake_news = pd.read_csv('Fake.csv')
        
        # Identify text column (could be 'text', 'title', or other)
        # Try common column names
        text_cols = ['text', 'title', 'content', 'article']
        true_text_col = None
        fake_text_col = None
        
        for col in text_cols:
            if col in true_news.columns:
                true_text_col = col
                break
        
        for col in text_cols:
            if col in fake_news.columns:
                fake_text_col = col
                break
        
        if true_text_col is None:
            true_text_col = true_news.columns[0]  # Use first column
        if fake_text_col is None:
            fake_text_col = fake_news.columns[0]  # Use first column
        
        print(f"Using column '{true_text_col}' from True.csv")
        print(f"Using column '{fake_text_col}' from Fake.csv")
        
        # Create combined dataset
        true_df = pd.DataFrame({
            'text': true_news[true_text_col],
            'label': 'real'
        })
        
        fake_df = pd.DataFrame({
            'text': fake_news[fake_text_col],
            'label': 'fake'
        })
        
        # Combine
        combined = pd.concat([true_df, fake_df], ignore_index=True)
        
        # Shuffle
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save
        combined.to_csv('dataset.csv', index=False)
        
        print(f"\n✓ Dataset created successfully!")
        print(f"  Total samples: {len(combined)}")
        print(f"  Real news: {len(combined[combined['label']=='real'])}")
        print(f"  Fake news: {len(combined[combined['label']=='fake'])}")
        print(f"  Saved to: dataset.csv")
        
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        print("\nMake sure True.csv and Fake.csv are in the current directory.")


if __name__ == "__main__":
    if os.path.exists('True.csv') and os.path.exists('Fake.csv'):
        prepare_dataset()
    else:
        download_kaggle_dataset()

