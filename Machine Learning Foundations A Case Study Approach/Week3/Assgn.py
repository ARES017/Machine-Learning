# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 22:51:05 2018

@author: Ananthu
"""

import graphlab
import graphlab.aggregate as agg

products = graphlab.SFrame('Course1/Week3/amazon_baby.gl/')

products.head()

selected_words = ['awesome','great','fantastic','amazing','love','horrible','bad','terrible','awful','wow','hate']
        
### Qn 1 & 2

products['word_count'] = graphlab.text_analytics.count_words(products['review']) ### 183531 rows

for i in range(len(selected_words)):
    def word_count(x):
        if selected_words[i] in x.keys():
            return x[selected_words[i]]
        else:
            return 0
    products[selected_words[i]] = products['word_count'].apply(lambda x:word_count(x))
    
selected_word_count = []
for i in range(len(selected_words)):
    selected_word_count.append(products[selected_words[i]].sum())

### Qn 3 & 4

products = products[products['rating'] != 3]
products['sentiment'] = products['rating'] >=4

train_data,test_data = products.random_split(.8, seed=0)

selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=selected_words,
                                                     validation_set=test_data)
                                                    
selected_words_model['coefficients'].sort('value')
                 
### Qn 5 & 6

selected_words_model.evaluate(test_data) ### accuracy - 0.8431419649291376
### accuracy of sentiment_model in ipython(lecture) - 0.916256305548883

### Qn 7 & 8

## Majority Class Classifier

num_positive  = (train_data['sentiment'] == +1).sum()
num_negative = (train_data['sentiment'] == 0).sum()                

test_num_positive  = (test_data['sentiment'] == +1).sum()
test_num_negative = (test_data['sentiment'] == 0).sum()
float(test_num_positive)/len(test_data)
     
### Qn 9 - 0.9 to 1
     
### Qn 10
     
baby_diaper_reviews = products[products['name'] == 'Baby Trend Diaper Champ']
baby_diaper_reviews['predicted_sentiment'] = selected_words_model.predict(baby_diaper_reviews, output_type='probability')
baby_diaper_reviews.sort('rating').tail()

### Qn 11

baby_diaper_reviews[-1]  # Option d