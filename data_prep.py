import pandas as pd
from sklearn.model_selection import train_test_split

import pandas as pd
from emotion_lbl import all_emotions
import numpy as np
import re
import string
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# train_df = pd.read_csv('final_project/archive 3/data/full_dataset/goemotions_1.csv')
# val_df = pd.read_csv('final_project/archive 3/data/full_dataset/goemotions_2.csv')
# test_df = pd.read_csv('final_project/archive 3/data/full_dataset/goemotions_3.csv')
dir1 = '/Users/zhankaiyan/PycharmProjects/CSCI-360/final_project/archive 3/data/full_dataset/goemotions_1.csv'
dir2 = '/Users/zhankaiyan/PycharmProjects/CSCI-360/final_project/archive 3/data/full_dataset/goemotions_2.csv'
dir3 = '/Users/zhankaiyan/PycharmProjects/CSCI-360/final_project/archive 3/data/full_dataset/goemotions_3.csv'

set1 = pd.read_csv(dir1)
set2 = pd.read_csv(dir2)
set3 = pd.read_csv(dir3)

# prepocessing the data
merge_df = pd.concat([set1, set2, set3], ignore_index=True)
# print(merge_df.shape)

pos_labels = ['admiration','approval', 'amusement', 'caring', 'desire',
              'excitement', 'gratitude', 'joy', 'love','optimism', 'pride', 'relief']
neg_labels = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
              'embarrassment','fear', 'grief', 'nervousness', 'remorse', 'sadness']
ambi_labels = ['confusion', 'curiosity', 'realization', 'surprise']
anger_labels = [ "anger", "annoyance", "disapproval", "disgust"]
fear_labels = ["fear", "nervousness"]
joy_labels = ["joy", "amusement", "approval", "excitement", "gratitude","love", "optimism", "relief", "pride", "admiration", "desire", "caring"]
sadness_labels = ["sadness", "disappointment", "embarrassment", "grief", "remorse"]
surprise_labels = ["surprise", "realization", "confusion", "curiosity"]

merge_df = merge_df.drop(columns=['author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id']) #  eliminate useless columns
merge_df = merge_df.dropna() # eliminate useless values
merge_df = merge_df[~merge_df['example_very_unclear']] # eliminate very unclear comments
# print(merge_df.shape)
# reshape the data
id_vars = ['text', 'id']
value_vars = merge_df.columns[4:]
merge_df = merge_df.melt(id_vars=id_vars, value_vars=value_vars, var_name='emotion', value_name='label')
merge_df = merge_df[merge_df['label'] != 0]

counts = merge_df.groupby(['id', 'emotion']).size().reset_index(name='count')
merge_df = pd.merge(merge_df, counts, on=['id', 'emotion'])
merge_df = merge_df.drop_duplicates(subset=['id', 'emotion'], keep='first')
merge_df = merge_df.drop(columns=['label'])
# first reorder by 3-type sentiments
# convert string expression to nums
pos_list = []
neg_list = []
ambi_list = []
for ele in pos_labels:
    pos_list.append(all_emotions.index(ele))
for ele in neg_labels:
    neg_list.append(all_emotions.index(ele))
for ele in ambi_labels:
    ambi_list.append(all_emotions.index(ele))

def get_sentiments(emotion_value):
    if any(_ in pos_list for _ in emotion_value):
        return 0
    if any(_ in neg_list for _ in emotion_value):
        return 1
    if any(_ in ambi_list for _ in emotion_value):
        return 2


merge_df['sentiment'] = merge_df['emotion'].apply(get_sentiments)
# then reorder by 6-type principle emotions
anger_list = []
fear_list = []
joy_list = []
sadness_list = []
surprise_list = []
# convert string expression to
for ele in anger_labels:
    anger_list.append(all_emotions.index(ele))
for ele in fear_labels:
    fear_list.append(all_emotions.index(ele))
for ele in joy_labels:
    joy_list.append(all_emotions.index(ele))
for ele in sadness_labels:
    sadness_list.append(all_emotions.index(ele))
for ele in surprise_labels:
    surprise_list.append(all_emotions.index(ele))


def get_emo(emotion_value):
    if any(_ in anger_list for _ in emotion_value):
        return 0
    elif any(_ in fear_list for _ in emotion_value):
        return 1
    elif any(_ in joy_list for _ in emotion_value):
        return 2
    elif any(_ in sadness_list for _ in emotion_value):
        return 3
    elif any(_ in surprise_list for _ in emotion_value):
        return 4
    else:
        return 5

merge_df['emo_group'] = merge_df['emotion'].apply(get_emo)
all_emotions_neutral_map = {value: index for index, value in enumerate(all_emotions)}
# print(all_emotions_neutral_map)
merge_df['emotion'] = merge_df['emotion'].apply(lambda x: all_emotions_neutral_map[x])
# print(merge_df)
data = merge_df # used for importng to other files
train_df, test_df = train_test_split(merge_df, test_size=0.85, random_state=42)






