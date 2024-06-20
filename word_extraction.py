from data_prep import merge_df
from emotion_lbl import all_emotions_neutral
import pandas as pd
from collections import Counter
import re

# define a function to count the frequency of words in a string
def count_words(text):
    # remove all punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # convert to lowercase
    text = text.lower()
    # split into words
    words = text.split()
    # count word frequency
    return Counter(words)

# group the dataframe by emotion and apply the count_words function to the text column
word_counts = merge_df.groupby('emotion')['text'].apply(' '.join).reset_index()
# print(len(word_counts.loc[1]))
# print(len(word_counts.loc[1][1]))
for _, row in word_counts.iterrows():
    emotion = row['emotion']
    text = row['text']
    print(count_words(text))



# print the top 10 words for each emotion
# for i, row in word_counts.iterrows():
#     emotion = row['emotion']
#     print(f"Top 10 words for {emotion}:")
#     for word, count in row['text'].most_common(10):
#         print(f"{word}: {count}")
