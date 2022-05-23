'''Here’s the list of reviews of Chrome apps - scraped from Playstore.  DataSet Link
Problem statement - There are times when a user writes Good, Nice App or any other positive text, in the review and gives 1-star rating. Your goal is to identify the reviews where the semantics of review text does not match rating. 
Your goal is to identify such ratings where review text is good, but rating is negative- so that the support team can point this to users. 

Deploy it using - Flask/Streamlit etc and share the live link.'''



# Importing Libraries
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Downloading Pre-trained Models
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = TFAutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_analysis = pipeline('sentiment-analysis', model = model, tokenizer = tokenizer)

# Sentiment Analysis of given dataset
def reviews(input_data):
    df = pd.read_csv(input_data)
    df = df[df.Star == 1]
    df.reset_index(drop=True, inplace=True)
    X = df.Text
    sentiment = []
    for i in range(len(X)):
        if sentiment_analysis(X[i])[0]['label'] == 'LABEL_2':
            sentiment.append(1)
        else:
            sentiment.append(0)
    df['sentiment'] = sentiment
    y = df[df.sentiment == 1]
    y.reset_index(drop=True, inplace=True)
    return y

# Deploying Using Streamlit
def main():
    st.title("Contradict Review Filter")
    input_data = st.file_uploader(label='Upload a CSV File', type=['csv'])
    
    if st.button('Filter Reviews'):
        if input_data is not None:
            Filtered_Reviews = reviews(input_data)
            st.success(st.write(Filtered_Reviews))
        else:
            st.markdown('### Upload a CSV file')

    
    

if __name__ == '__main__':
    main()
