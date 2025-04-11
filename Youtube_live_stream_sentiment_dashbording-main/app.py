import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import time
import plotly.express as px
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pickle
from wordcloud import WordCloud
from nltk.corpus import stopwords
# from transformers import pipeline

stopwords_list=stopwords.words('english')


st.set_page_config(
    page_title="Real-Time Sentiment Dashboard",
    page_icon="📺",
    layout="wide",
)


# Load the pipeline from the pickle file
pipeline = pickle.load(open("stacking_model.pkl", "rb"))

# classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)


# Ensure you have downloaded the necessary NLTK data files
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text(text):

    # Remove hyperlinks
    hyperlink_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = re.sub(hyperlink_pattern, '', text)

    # Tokenize the text
    tokens = word_tokenize(text.lower())
   
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

def clean_text(text):
    stemmer = PorterStemmer()
    hyperlink_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = re.sub(hyperlink_pattern, '', text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)
    
emotional_categories=['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    
    # Make prediction
    predicted_label = pipeline.predict([cleaned_text])
    label = emotional_categories[predicted_label[0]]
    
    # Get the probability of the predicted label
    probabilities = pipeline.predict_proba([cleaned_text])
    score= np.max(probabilities)

    #predicted_emotion = classifier(input_text)
    #prediction_data = predicted_emotion[0]
    
    # Calculate maximum score and corresponding label
    # score = 0
    # label = 'x'

     
    # for x in prediction_data:
    #     if x['score'] > score:
    #         score = x['score']
    #         label = x['label']

    return label,score



# Set up your API key
API_KEY = 'AIzaSyDG3hymqtF_Ak12_Jo1yjUe2YSimwjmXpI'

# Set up the YouTube API service
youtube = build('youtube', 'v3', developerKey=API_KEY)

def analyze_sentiment(comment_text):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    comment_text=preprocess_text(comment_text)
    sentiment_scores = sentiment_analyzer.polarity_scores(comment_text)
    # Return compound sentiment score
    return sentiment_scores['compound']

def get_live_chat_id(video_url):
    # Parse the video ID from the video URL
    parsed_url = urlparse(video_url)
    video_id = parse_qs(parsed_url.query).get('v')
    if video_id:
        video_id = video_id[0]
    else:
        return None
    
    # Get live chat ID of the live stream
    response = youtube.videos().list(
        part='liveStreamingDetails',
        id=video_id
    ).execute()

    if 'items' in response and response['items']:
        live_chat_id = response['items'][0]['liveStreamingDetails']['activeLiveChatId']
        return live_chat_id
    else:
        return None

def scrape_live_comments(video_url,placeholder, sentiment_chart,
                         prev_total_comments, prev_positive_comments, prev_negative_comments, prev_neutral_comments):
    live_chat_id = get_live_chat_id(video_url)
    if live_chat_id:
        total_comments = 0
        positive_comments = 0
        negative_comments = 0
        neutral_comments = 0
        anger_value,fear_value, joy_value, love_value, sadness_value, surprise_value=0, 0, 0, 0, 0, 0
        pre_anger, pre_fear, pre_joy, pre_love, pre_sadness, pre_surprise=0, 0, 0, 0, 0, 0

        next_page_token = None
        data = []
        
        while True:
            response = youtube.liveChatMessages().list(
                liveChatId=live_chat_id,
                part='snippet',
                pageToken=next_page_token,
                maxResults=100 # Adjust as needed
            ).execute()

            for item in response['items']:
                if 'snippet' in item:
                    comment_text = item['snippet']['displayMessage']
                    sentiment = analyze_sentiment(comment_text)
                    total_comments += 1
                    if sentiment > 0:
                        positive_comments += 1
                        sentiment_label = 'Positive'
                    elif sentiment < 0:
                        negative_comments += 1
                        sentiment_label = 'Negative'
                    else:
                        neutral_comments += 1
                        sentiment_label = 'Neutral'
                    
                    emotion,score=predict_emotion(comment_text)

                    if(emotion == 'anger'):
                        anger_value+=1

                    elif(emotion == 'fear'):
                        fear_value+=1
                    
                    elif(emotion == 'joy'):
                        joy_value+=1
                    
                    elif(emotion == 'love'):
                        love_value+=1

                    elif(emotion == 'sadness'):
                        sadness_value+=1
                    
                    else:
                        surprise_value+=1
                    

                    data.append({'Comment': comment_text, 'Sentiment': sentiment_label,'Emotion':emotion,'Emotion_Confidence':score})

            
            # Calculate delta values
            dataframe=pd.DataFrame(data)
            
            delta_total_comments = total_comments - prev_total_comments
            delta_positive_comments = positive_comments - prev_positive_comments
            delta_negative_comments = negative_comments - prev_negative_comments
            delta_neutral_comments = neutral_comments - prev_neutral_comments

            delta_anger=anger_value-pre_anger
            delta_joy=joy_value-pre_joy
            delta_fear=fear_value-pre_fear
            delta_love=love_value-pre_love
            delta_sadness=sadness_value-pre_sadness
            delta_surprise=surprise_value-pre_surprise


            
            # Update KPIs
            with placeholder.container():
                st.markdown("##### sentiments")
                total_comments_placeholder, positive_comments_placeholder, negative_comments_placeholder,neutral_comments_placeholder= st.columns(4)
                total_comments_placeholder.metric(label="Total Comments", value=total_comments, delta=delta_total_comments)
                positive_comments_placeholder.metric(label="Positive", value=positive_comments, delta=delta_positive_comments)
                negative_comments_placeholder.metric(label="Negative", value=negative_comments, delta=delta_negative_comments)
                neutral_comments_placeholder.metric(label="Neutral", value=neutral_comments, delta=delta_neutral_comments)

            with placeholder1.container():
                st.markdown("##### Emotions")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Place each metric and placeholder in its respective column
                with col1:
                    st.metric(label='Anger', value=anger_value, delta=delta_anger)
                    st.empty()  # Placeholder
                with col2:
                    st.metric(label='Fear', value=fear_value, delta=delta_fear)
                    st.empty()  # Placeholder
                with col3:
                    st.metric(label='Joy', value=joy_value, delta=delta_joy)
                    st.empty()  # Placeholder
                with col4:
                    st.metric(label='Love', value=love_value, delta=delta_love)
                    st.empty()  # Placeholder
                with col5:
                    st.metric(label='Sadness', value=sadness_value, delta=delta_sadness)
                    st.empty()  # Placeholder
                with col6:
                    st.metric(label='Surprise', value=surprise_value, delta=delta_surprise)
                    st.empty()  # Placeholder



            # Bar Chart
            df = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                'Count': [positive_comments, negative_comments, neutral_comments]
            })

            df_percent = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                'Percentage': [positive_comments/total_comments*100, negative_comments/total_comments*100, neutral_comments/total_comments*100]
            })

            df_emotion = pd.DataFrame({
                'Emotion' : ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise'],
                'Percentage':[anger_value/total_comments*100,fear_value/total_comments*100,joy_value/total_comments*100,love_value/total_comments*100,sadness_value/total_comments*100,surprise_value/total_comments*100]
            })

            
    # Plot bar chart using Plotly
            with sentiment_chart.container():
                fig_col1, fig_col2 = st.columns(2)
                with fig_col1:
                    st.markdown("### Sentiment Analysis")
                    # fig = px.bar(df, x='Sentiment', y='Count', text='Count', title='Sentiment Analysis', 
                    # labels={'Count': 'Count', 'Sentiment': 'Sentiment'})
                    # fig_col1.write(fig)
                    fig_percent = px.bar(df_percent, x='Sentiment', y='Percentage', text='Percentage', title='Sentiment Distribution (Percentage)', 
                    labels={'Percentage': 'Percentage', 'Sentiment': 'Sentiment'})
                    fig_percent.update_traces(texttemplate='%{text}%', textposition='outside')
                    fig_col1.write(fig_percent)


                with fig_col2:
                    st.markdown("### Emotion Analysis")
                    fig_emotions = px.bar(df_emotion, x='Emotion', y='Percentage', text='Percentage', title='Emotion Distribution (Percentage)', 
                    labels={'Percentage': 'Percentage', 'Emotion': 'Emotion'})
                    fig_emotions.update_traces(texttemplate='%{text}%', textposition='outside')
                    fig_col2.write(fig_emotions)

            datset_display.write(dataframe)

             
            # Update previous values
            prev_total_comments = total_comments
            prev_positive_comments = positive_comments
            prev_negative_comments = negative_comments
            prev_neutral_comments = neutral_comments

            pre_anger=anger_value
            pre_fear=fear_value
            pre_joy=joy_value
            pre_love=love_value
            pre_sadness=sadness_value
            pre_surprise=surprise_value
            
            time.sleep(10)
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
    else:
        st.error("Live chat not available for this video.")

# Streamlit App
st.title("📺 Real-Time Sentiment Dashboard")

st.sidebar.title("Navigation")
st.sidebar.image("youtube-5702871_1280-removebg-preview.png", use_column_width=True)
st.sidebar.subheader("Anaylse the Sentiments and Emotions of the Live Stream of the  Video ")
video_url = st.sidebar.text_input("Paste the link of the video")

placeholder = st.empty()
placeholder1 = st.empty()

sentiment_chart = st.empty()
datset_display=st.empty()



prev_total_comments = prev_positive_comments = prev_negative_comments = prev_neutral_comments = 0

if st.sidebar.button("Analyze Comments"):
    scrape_live_comments(video_url,placeholder, sentiment_chart,
                         prev_total_comments, prev_positive_comments, prev_negative_comments, prev_neutral_comments)
