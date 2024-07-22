from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
extractor=URLExtract()
import emoji
from nltk.stem.porter import PorterStemmer
import re
import pickle


def fetch_stats(selected_user,df):
    if selected_user !='Overall':
        df=df[df['User']==selected_user]
    # fetch number of messages
    num_messages=df.shape[0]
    # fetch number of Words    
    words=[]
    for message in df['Message']:
        words.extend(message.split())

    # fetch number of Media File
    num_media=df[df['Message']=='<Media omitted>\n'].shape[0]

    # fetch number of Link
    Link=[]
    for message in df['Message']:
        Link.extend(extractor.find_urls(message))


    return num_messages,len(words),num_media,len(Link)
   # if selected_user=='Overall':
        #fetch number of messages
      #  num_messages= df.shape[0]
     #  words = []
     # for message in df['Message']:
        #    words.extend(message.split())
     #   return num_messages,len(words)
    #else:
      #  new_df=df[df['User']==selected_user]
       # num_messages=new_df.shape[0]
       # words = []
      #  for message in new_df['Message']:
      #      words.extend(message.split())
      #  return num_messages,len(words)

def most_busy_users(df):
    x=df['User'].value_counts().head()
    new_df=round((df['User'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns={'User':'Name','count':'Percentage'})
    return x,new_df

def creat_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user !='Overall':
        df=df[df['User']==selected_user]

    # Group Notifications removed
    temp = df[df['User'] != 'group_notifications']

    # Media Omitted Removed
    temp = temp[temp['Message'] != '<Media omitted>\n']

        # Stop Words Removed
    def remove_stop_words(message):
        y=[]
        for word in message.lower().split():
            if word not in stop_words:
             y.append(word)
        return " ".join(y)

    wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['Message']= temp['Message'].apply(remove_stop_words)
    df_wc=wc.generate(temp['Message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):
    if selected_user !='Overall':
        df=df[df['User']==selected_user]
    f=open('stop_hinglish.txt','r')
    stop_words=f.read()

    # Group Notifications removed
    temp = df[df['User'] != 'group_notifications']

    # Media Omitted Removed
    temp = temp[temp['Message'] != '<Media omitted>\n']

    #Stop Words Removed
    words = []
    for message in temp['Message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df=pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df=df[df['User']==selected_user]

    emojis=[]
    for message in df['Message']:
        emojis.extend(c for c in message if c in emoji.EMOJI_DATA)

    emojis_df=pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emojis_df

def monthly_timeline(selected_user,df):
    if selected_user!='Overall':
        df=df[df['User']==selected_user]

    timeline=df.groupby(['Year','Month_num','Month']).count()['Message'].reset_index()
    time=[]
    for i in range(timeline.shape[0]):
        time.append(timeline['Month'][i]+"-"+str(timeline['Year'][i]))
    timeline['Time']=time
    timeline = timeline.drop(['Year', 'Month_num', 'Month'], axis=1)
    return timeline

def daily_timeline(selected_user,df):
    if selected_user!='Overall':
        df=df[df['User']==selected_user]

    daily_timeline = df.groupby('_Date_').count()['Message'].reset_index()
    return daily_timeline

def weak_activity_map(selected_user,df):
    if selected_user!='Overall':
        df=df[df['User']==selected_user]

    return df['Day_name'].value_counts()

def month_activity_map(selected_user,df):
    if selected_user!='Overall':
        df=df[df['User']==selected_user]

    return df['Month'].value_counts()

def activity_heatmap(selected_user,df):
    if selected_user!='Overall':
        df=df[df['User']==selected_user]

    activity_heatmap=df.pivot_table(index='Day_name',columns='Period',values='Message',aggfunc='count').fillna(0)
    return activity_heatmap

def sentiment(selected_user,df):
    if selected_user!='Overall':
        df=df[df['User']==selected_user]
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    # Group Notifications removed
    temp = df[df['User'] != 'group_notifications']

    # Media Omitted Removed
    temp = temp[temp['Message'] != '<Media omitted>\n']

    stemmer = PorterStemmer()
    emoji_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')

    def preprocessing(Message):
        Message = re.sub(r'<[^>]*>', ' ', Message)

        # Extract Emojies
        emojis = emoji_pattern.findall(Message)

        # Replace any none alphabet Characters with space
        Message = re.sub(r'[^a-zA-Z]', ' ', Message)

        Message = Message.lower().split()

        # Remove Stopwords
        Message = [stemmer.stem(word) for word in Message if word not in stop_words]
        Message = ' '.join(Message)
        Message += ' ' + ' '.join(emojis)
        return "".join(Message)

    temp['Message'] = temp['Message'].apply(lambda x: preprocessing(x))

    predictor = pickle.load(open('lr.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))

    def prediction(comment):
        preprocessed_comment = preprocessing(comment)
        comment_list = [preprocessed_comment]  # Wrap the preprocessed comment in a list
        comment_vector = tfidf.transform(comment_list)
        prediction = predictor.predict(comment_vector)[0]
        return prediction

    temp["Prediction"] = temp['Message'].apply(lambda x: prediction(x))

    Sentiment = []

    # Iterate through the predictions
    for prediction in temp['Prediction']:
        if prediction == 1:
            Sentiment.append('Positive')
        else:
            Sentiment.append('Negative')

    # Add the Sentiment list as a new column in the DataFrame
    temp['Sentiment'] = Sentiment

    Analysis = temp['Sentiment'].value_counts().reset_index()

    return Analysis
















