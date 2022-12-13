import wordcloud as wc
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image 
import random
import cleanData


df=pd.read_csv("train.csv")
df.head()

# Groupby by country
label = df.groupby("Label")

for title, group in label :
    print (title)
    
print(label.describe().head())

business = label.get_group("Business")
entertainment = label.get_group("Entertainment")
health = label.get_group("Health")
tech = label.get_group("Technology")

categories = [business , entertainment ,health ,  tech]

images = ["images/wd_mask_1.png" , "images/wd_mask_2.jpg" , "images/wd_mask_3.jpg" , "images/wd_mask_4.jpg"]

i=1

for index , category in enumerate(categories) :

    text=""
    
    print(i)
    i+=1
    
    
    for title in category["Title"]:
        text+=str(title)+ " "
    for doc in category["Content"]:
        text+=str(doc) + " "
    print(len(text))   
    txt=cleanData.DataCleaner(text)
    text=txt.clean()
    print(len(text))
    
    char_mask = np.array(Image.open(images[index]))    
    image_colors = wc.ImageColorGenerator(char_mask)
    
    cloud = wc.WordCloud(background_color='black' , mask=char_mask , max_words=95).generate(text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    #plt.show()

    cloud.to_file("wordcloud" + str(index) + ".png")

