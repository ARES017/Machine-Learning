# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 22:50:12 2018

@author: Ananthu
"""

import graphlab

people = graphlab.SFrame('Course1/Week4/people_wiki.gl/')

people.head()

### Qn 1

elton_john = people[people['name'] == 'Elton John']
elton_john['text']

elton_john['word_count'] = graphlab.text_analytics.count_words(elton_john['text'])
elton_word_count_table = elton_john[['word_count']].stack('word_count', new_column_name = ['word','count'])
elton_word_count_table.sort('count',ascending=False)

### Qn 2

people['word_count'] = graphlab.text_analytics.count_words(people['text'])
tfidf = graphlab.text_analytics.tf_idf(people['word_count'])
people['tfidf'] = tfidf
elton_john = people[people['name'] == 'Elton John']
elton_john[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)

### Qn 3

victoria = people[people['name'] == 'Victoria Beckham']
graphlab.distances.cosine(elton_john['tfidf'][0],victoria['tfidf'][0])

### Qn 4 & 5

paul = people[people['name'] == 'Paul McCartney']
graphlab.distances.cosine(elton_john['tfidf'][0],paul['tfidf'][0])

### Qn 6 & 8

knn_model = graphlab.nearest_neighbors.create(people,features=['word_count'],label='name',distance='cosine')
knn_model.query(elton_john)
knn_model.query(victoria)


### Qn 7 & 9

knn_model_tfidf = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name',distance='cosine')
knn_model_tfidf.query(elton_john)
knn_model_tfidf.query(victoria)
