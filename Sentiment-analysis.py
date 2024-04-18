import requests
import pandas as pd
import json
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

if len(sys.argv) > 2:
    query = sys.argv[1]
    key = sys.argv[2]
else:
    print("Wrong arguments")
    sys.exit(1)

url = "https://twitter-api45.p.rapidapi.com/search.php"
querystring = {"query":query,"search_type":"Top"}

headers = {
	"X-RapidAPI-Key": key, #parse you API key
	"X-RapidAPI-Host": "twitter-api45.p.rapidapi.com"
}
response = requests.get(url, headers=headers, params=querystring)
data = json.loads(response.text)

simplified_data = []
for item in data['timeline']:
    simplified_entry = {
        'tweet_id': item['tweet_id'],
        'screen_name': item['screen_name'],
        'favorites': item['favorites'],
        'created_at': item['created_at'],
        'text': item['text'],
        'lang': item['lang'],
    }
    simplified_data.append(simplified_entry)

df = pd.DataFrame(simplified_data)
df = df[df['lang'] == 'en']

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
Sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

result = []
for index, row in df.iterrows():
    result += Sentiment_analysis(row['text']) 
resultdf = pd.DataFrame(result)

sentiment_counts = resultdf['label'].value_counts()
total_rows = len(resultdf)
sentiment_percentages = (sentiment_counts / total_rows) * 100

fig, ax = plt.subplots()
ax.pie(sentiment_percentages, labels=sentiment_percentages.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal') 

plt.title('Sentiment Analysis Results')
plt.show()