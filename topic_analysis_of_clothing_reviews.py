"""
Project Instructions

    Create and store the embeddings
        Embed the reviews using a suitable text embedding algorithm and store them as list in the variable embeddings.
    Dimensionality reduction & visualization
        Apply an appropriate dimensionality reduction technique to reduce the embeddings to a 2-dimensional numpy array and store this array in the variable embeddings_2d.
        Then, use this variable to plot a 2D visual representation of the reviews. 
    Feedback categorization
        Use your embeddings to identify some reviews that discuss topics such as 'quality', 'fit', 'style', 'comfort', etc.
    Similarity search function
        Write a function that outputs the closest 3 reviews to a given input review, enabling a more personalized customer service response.
        Apply this function to the first review "Absolutely wonderful - silky and sexy and comfortable", and store the output as a list in the variable most_similar_reviews.
"""

# Run this cell to install ChromaDB if desired
try:
    assert version('chromadb') == '0.4.17'
except:
    !pip install chromadb==0.4.17
try:
    assert version('pysqlite3') == '0.5.2'
except:
    !pip install pysqlite3-binary==0.5.2
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

# Load the dataset
import pandas as pd
reviews = pd.read_csv("womens_clothing_e-commerce_reviews.csv")

# Display the first few entries
reviews.head()

