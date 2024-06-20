import pandas as pd
import numpy as np
from emotion_lbl import all_emotions_neutral # 数组格式  type: list
import matplotlib.pyplot as plt
import seaborn as sns

dir1 = '/Users/zhankaiyan/PycharmProjects/CSCI-360/final_project/archive 3/data/full_dataset/goemotions_1.csv'
dir2 = '/Users/zhankaiyan/PycharmProjects/CSCI-360/final_project/archive 3/data/full_dataset/goemotions_2.csv'
dir3 = '/Users/zhankaiyan/PycharmProjects/CSCI-360/final_project/archive 3/data/full_dataset/goemotions_3.csv'
set1 = pd.read_csv(dir1)
set2 = pd.read_csv(dir2)
set3 = pd.read_csv(dir3)

merge_df = pd.concat([set1, set2, set3], ignore_index=True)

# print(merge_df)

# ---------------------------------

print(merge_df.shape)
print(merge_df['example_very_unclear'].sum())
temp_merge_df = merge_df[~merge_df['example_very_unclear']]
print(temp_merge_df.shape)

# test of multiple labels
# calculate the average ratings of one comments and calculate the means
print("%d Examples" % (len(set(merge_df["id"]))))
print("%d Annotations" % len(merge_df))
print("%d unique raters" % len(merge_df["rater_id"].unique()))
print("%.3f marked unclear" %
      (merge_df["example_very_unclear"].sum() / len(merge_df)))

ata = merge_df[merge_df[all_emotions_neutral].sum(axis=1) != 0]

print("Distribution of number of labels per example:")
print(merge_df[all_emotions_neutral].sum(axis=1).value_counts() / len(merge_df))
print("%.2f with more than 3 labels" %
      ((merge_df[all_emotions_neutral].sum(axis=1) > 3).sum() /
       len(merge_df)))  # more than 3 labels

print("Label distributions:")
print((merge_df[all_emotions_neutral].sum(axis=0).sort_values(ascending=False) /
       len(merge_df) * 100).round(2))
ratings = merge_df.groupby("id")[all_emotions_neutral].mean()
#
# # Compute the correlation matrix


# plotting
fig = plt.subplots(figsize=(6,8))
# cmap = sns.diverging_palette(220, 10, as_cmap=True)

# sns.heatmap(
#     corr,
#     cbar=True,
#     square=True,
#     cmap='coolwarm',
#     vmax=0.3,
#     vmin=-0.3,
#     center=0,
#     linewidths=.5,
#     cbar_kws={"shrink": .5})
# plt.xlabel('segment emotions')
# plt.ylabel('segment enotions')
# plt.title('correlations in principle variables')
# plt.show()

# group_by emotions
pos_labels = ['admiration','approval', 'amusement', 'caring', 'desire',
              'excitement', 'gratitude', 'joy', 'love','optimism', 'pride', 'relief']
neg_labels = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
              'embarrassment','fear', 'grief', 'nervousness', 'remorse', 'sadness']
ambi_labels = ['confusion', 'curiosity', 'realization', 'surprise']

emotion_count = {}
sentiment_count = {}

pos_sum = 0
for label in pos_labels:
    df = merge_df[merge_df[label]==1]
    emotion_count[label] = len(df)
    pos_sum += len(df)
sentiment_count['positive'] = pos_sum

neg_sum = 0
for label in neg_labels:
    df = merge_df[merge_df[label]==1]
    emotion_count[label] = len(df)
    neg_sum += len(df)
sentiment_count['negative'] = neg_sum

ambi_sum = 0
for label in ambi_labels:
    df = merge_df[merge_df[label]==1]
    emotion_count[label] = len(df)
    ambi_sum += len(df)

df = merge_df[merge_df['neutral']==1]
emotion_count['neutral'] = len(df)
sentiment_count['neutral']  = len(df)
sentiment_count['ambiguous'] = ambi_sum
print(emotion_count)

# plot bar
# Univariate distribution
# Convert the dictionary to a pandas DataFrame
rdf = pd.DataFrame.from_dict(emotion_count, orient='index').reset_index()
# Rename the columns to more descriptive names
rdf.rename(columns={'index': 'Emotion', 0: 'Count'}, inplace=True)
# Sort the DataFrame by count in descending order
rdf.sort_values(by=['Count'], ascending=False, inplace=True)
# Create a bar plot using matplotlib
fig, ax = plt.subplots()
ax.bar(rdf['Emotion'], rdf['Count'], color='blue')
# Set the labels of the x-ticks to be at a 45 degree angle and aligned to the right
plt.xticks(rotation=45, ha='right')
# Set the bottom margin of the plot to be 0.25
plt.gcf().subplots_adjust(bottom=0.25)
# Set the title of the plot
plt.title('Emotion Counts')
# Set the labels for the x- and y-axes
plt.xlabel('Emotion')
plt.ylabel('Count')
# Display the plot
plt.show()

# pie chart
df = pd.DataFrame.from_dict(sentiment_count, orient="index")
df.rename(columns={"index": "Sentiment", 0: "Count"}, inplace=True)
df.plot.pie(y="Count",autopct='%1.1f%%',
            colors=["#7080E2", "#EF4135", "#FAEBD7","#0DC9B6"], explode=(0, 0, 0.1, 0), fontsize=16,
            labels=["Positive", "Negative", "Neutral", "Ambiguous"],
           legend=False, ylabel="",)

plt.show()

# segmented emotional analysis
anger_list = [ "anger", "annoyance", "disapproval", "disgust"]
fear_list = ["fear", "nervousness"]
joy_list = ["joy", "amusement", "approval", "excitement", "gratitude","love", "optimism", "relief", "pride", "admiration", "desire", "caring"]
sadness_list = ["sadness", "disappointment", "embarrassment", "grief", "remorse"]
surprise_list = ["surprise", "realization", "confusion", "curiosity"]

def label_emotion_boraded(row):
    if row['Emotion'] in anger_list:
        return "Anger"
    if row['Emotion'] in fear_list:
        return "Fear"
    if row['Emotion'] in joy_list:
        return "Joy"
    if row['Emotion'] in sadness_list:
        return "Sadness"
    if row['Emotion'] in surprise_list:
        return "Surprise"
    if row['Emotion'] == 'neutral':
        return "Neutral"
df = rdf
df["Broader Emotions"] = df.apply(lambda row: label_emotion_boraded(row), axis = 1)
broader_emotion_count = df.groupby("Broader Emotions").sum().reset_index()
broader_emotion_count.plot.pie(y = "Count",autopct='%1.1f%%', colors = sns.color_palette("Set2"), labels = broader_emotion_count["Broader Emotions"],
                              legend = False,  ylabel = "Count of data points", fontsize = 16)
plt.show()

#heatmap
corr = ratings.corr()
sns.set_theme(style="white")
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask = mask, square = True, linewidths = 1, xticklabels=True, yticklabels=True, robust = True)
plt.show()
