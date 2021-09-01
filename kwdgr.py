import streamlit as st 
from polyfuzz import PolyFuzz
import seaborn as sns
import base64
import csv
import sys
import pandas as pd
import numpy as np
from textblob import TextBlob
from google.colab import files
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
  

def get_categories(searched_nouns, keywords, n_categories=5):
    """
    return n highest weighted keywords from search terms as tuple
    """

    matches = []

    for keyword in keywords.itertuples():
        if len(matches) == n_categories:
            break

        try:
            if keyword.Index in searched_nouns:
                matches.append(keyword.Index)
        except:
            continue

    missing_count = abs(len(matches) - n_categories)

    for _ in range(missing_count):
        matches.append(None)

    return matches
    
def get_nouns(search_terms):
    """
    nltk blob tags
    NN: Noun, singular or mass
    NNS: Noun, plural
    NNP: Proper noun, singular Phrase
    NNPS: Proper noun, plural
    """

    nouns = None
    try:
        blob = TextBlob(search_terms)
        nouns = [
            noun[0]
            for noun in blob.tags
            if noun[1] in ("NN", "NNS", "NNP", "NNPS")
        ]
    except:
        pass

    return nouns
    
def get_nouns_phrases(search_terms):
    nouns = None
    try:
        nouns = TextBlob(search_terms).noun_phrases[0]
        nouns = tuple(nouns.split())
    except:
        pass

    return nouns
 
def main(input_file, output_file):
  df = pd.read_excel(input_file)

  # clean column names
  df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

  # df homogenize keywords column - currently mixed types
  # make all strings
  df.keywords = df.keywords.astype(str)

  # row wise -- if no nouns it will be empty
  df["nouns"] = df.apply(lambda row: get_nouns(row.keywords), axis=1)

  # duplicate record per noun for frequency count
  df_exploded = df.explode("nouns")

  # TODO: before applying weights, map different keywords that mean the same thing to the unifying keyword
  # Highest to lowest
  weighted_keywords = df_exploded.groupby(
      ["nouns"]
  ).keywords.agg(["count"]).sort_values(by=["count"], ascending=False)

  df['cat_1'], df['cat_2'], df['cat_3'], df['cat_4'], df['cat_5'] = [None, None, None]
  df['cat_1'], df['cat_2'], df['cat_3'], df['cat_4'], df['cat_5'] = df.apply(
       lambda row, weighted_keywords=weighted_keywords: get_categories(row.keywords, weighted_keywords),
       lambda row: get_categories(row.keywords, weighted_keywords),
       axis=1
   )
  df['results'] = df.apply(
      lambda row, weighted_keywords=weighted_keywords: get_categories(row.keywords, weighted_keywords),
      lambda row: get_categories(row.nouns, weighted_keywords),
      axis=1
  )

  df[
      ['cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5']
  ] = pd.DataFrame(df.results.values.tolist(), index=df.index)


  df.to_excel(output_file)
    
df = pd.read_excel("/content/kwd.xlsx")

df.head()

# clean column names
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

# df homogenize keywords column - currently mixed types
# make all strings
df.keywords = df.keywords.astype(str)

# row wise -- if no nouns it will be empty
df["nouns"] = df.apply(lambda row: get_nouns(row.keywords), axis=1)

# duplicate record per noun for frequency count
df_exploded = df.explode("nouns")

# TODO: before applying weights, map different keywords that mean the same thing to the unifying keyword
# Highest to lowest
weighted_keywords = df_exploded.groupby(
["nouns"]
).keywords.agg(["count"]).sort_values(by=["count"], ascending=False)

# df['cat_1'], df['cat_2'], df['cat_3'] = [None, None, None]
# df['cat_1'], df['cat_2'], df['cat_3'] = df.apply(
#     # lambda row, weighted_keywords=weighted_keywords: get_categories(row.keywords, weighted_keywords),
#     lambda row: get_categories(row.keywords, weighted_keywords),
#     axis=1
# )
df['results'] = df.apply(
# lambda row, weighted_keywords=weighted_keywords: get_categories(row.keywords, weighted_keywords),
lambda row: get_categories(row.nouns, weighted_keywords),
axis=1
)

df[
['cat_1', 'cat_2', 'cat_3','cat_4','cat_5' ]
] = pd.DataFrame(df.results.values.tolist(), index=df.index)

df.head()

df.to_csv("test_output.csv",index  = False)
