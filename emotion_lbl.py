import pandas as pd
dir = '/Users/zhankaiyan/PycharmProjects/CSCI-360/final_project/archive 3/data/emotions.txt'

emotion_df = pd.read_csv(dir)
header = ['emotions']
emotion_df.columns = header
# print(emotion_df['emotions'])

with open(dir, "r") as f:
    all_emotions = f.read().splitlines()
all_emotions_neutral = all_emotions + ['neutral']


if __name__ == '__main__':

    print(all_emotions_neutral)
    #
    # # Load the data
    # data = pd.read_csv('goemotions_1.csv')
    #
    # # Extract the emotion labels
    # emotion_labels = data.columns[2:].tolist()
    #
    # # Group the data by emotion and calculate the mean for each label
    # mean_labels = data.groupby('label')[emotion_labels].mean()
    #
    # # Calculate the correlation matrix between emotion labels
    # corr_matrix = mean_labels.corr(method='pearson')
    #
    # print(corr_matrix)


    # li = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire',
    # 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
    # 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral',
    # 'neutral']