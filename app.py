import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns


st.sidebar.title('WhatsApp Chat Analyzer')
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    data=bytes_data.decode(encoding="utf-8")
    df=preprocessor.preprocess(data)



    #fetch Unique Users From user Column
    user_list=df['User'].unique().tolist()
    user_list.remove('group_notifications')
    user_list.sort()
    user_list.insert(0,'Overall')
    selected_user=st.sidebar.selectbox("Show Analysis Wrt",user_list)
    if st.sidebar.button("Show Analysis"):
        #Stats Area
        num_messages,words,num_media,Link=helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4=st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Total Media")
            st.title(num_media)
        with col4:
            st.header("Total Links")
            st.title(Link)

        #Monthly timeline
        st.title("Monthly Timeline")
        timeline=helper.monthly_timeline(selected_user,df)
        fig,ax=plt.subplots()
        ax.plot(timeline['Time'],timeline['Message'],color='green')
        plt.xticks(rotation='vertical')

        col1,col2=st.columns(2)
        with col2:
            st.dataframe(timeline)
        with col1:
            st.pyplot(fig)

        #Daily Timeline
        st.title("Daily Timeline")
        daily_timeline=helper.daily_timeline(selected_user,df)
        fig,ax=plt.subplots()
        ax.plot(daily_timeline['_Date_'],daily_timeline['Message'],color='black')
        plt.xticks(rotation='vertical')

        col1,col2=st.columns(2)
        with col1:
            st.pyplot(fig)
        with col2:
            st.dataframe(daily_timeline)

        #Activity Map
        st.title("Activity Map")
        col1,col2=st.columns(2)
        # Most Busy Day
        with col1:
            st.header('Most Busy Day')
            busy_day=helper.weak_activity_map(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header('Most Busy Month')
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Activity HeatMap
        st.title("Activity Heatmap")
        activity_heatmap=helper.activity_heatmap(selected_user,df)
        fig,ax=plt.subplots()
        ax=sns.heatmap(activity_heatmap)
        st.pyplot(fig)






        # Finding the Busiest Users in the Group
        if selected_user=='Overall':
            st.title("Most Busy Users")
            x,new_df=helper.most_busy_users(df)
            fig,ax=plt.subplots()
            plt.xticks(rotation='vertical')

            col1,col2=st.columns(2)
            with col1:
                ax.bar(x.index,x.values)
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)



        #WordCloud
        st.title("WordCloud")
        df_wc = helper.creat_wordcloud(selected_user,df)
        fig,ax=plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)


        #Most Common Words
        most_common_df =helper.most_common_words(selected_user,df)
        fig,ax=plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title("Most Common Words")
        st.pyplot(fig)


        #Emoji Extractor
        emoji_df = helper.emoji_helper(selected_user, df)

        # Streamlit app layout
        st.title("Emoji Analysis")
        col1, col2 = st.columns(2)

        # Display emoji analysis DataFrame in the first column
        with col1:
            st.dataframe(emoji_df)

        # Plot pie chart with emojis in the second column
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1], labels=emoji_df[0], autopct="%0.2f", startangle=140)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax.set_title('Emoji Distribution')

            # Display the plot in Streamlit
            st.pyplot(fig)


        #Sentimental Analysis
        Analysis=helper.sentiment(selected_user,df)
        st.title("Sentiment Analysis")
        fig,ax=plt.subplots()
        ax.pie(Analysis['count'],labels=Analysis['Sentiment'],autopct="%0.2f", startangle=140)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_title('Percentage of Positive and Negative Sentiments  ')

        # Display the plot in Streamlit
        st.pyplot(fig)










