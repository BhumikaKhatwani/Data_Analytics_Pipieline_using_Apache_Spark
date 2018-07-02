import json
from datetime import timedelta, date
import requests
from bs4 import BeautifulSoup
import urllib
import time
import io
  

news1=[]
for page in range(0,5):
        url_value = "http://api.nytimes.com/svc/search/v2/articlesearch.json?q=politics&api-key=656fc558af8443c5a2327b391229db9b&sort=newest&page=" +str(page)
        time.sleep(1)
        print(url_value)
        #r = urllib.request.urlopen(url_value).read()
        r = requests.get(url_value)
        data = r.json()
        len(data["response"]["docs"])
        
        j=0
        for i in data['response']['docs']:
            url = i['web_url']
            r = requests.get(url).text
            time.sleep(1)
            soup = BeautifulSoup(r,'html.parser')
            
            title = soup.title
            article_paragraphs = soup.find_all('p')
            article = ""
            for p in article_paragraphs:
                article=article+p.get_text()
            file_name = "politics"+str(page)+str(j)+".txt"
            with open(file_name, "w") as ou:
                ou.write(article.encode('utf-8'))
            j=j+1
            news1.append(article)
len(news1)


