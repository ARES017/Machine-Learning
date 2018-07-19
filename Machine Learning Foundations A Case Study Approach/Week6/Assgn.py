# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 00:20:26 2018

@author: Ananthu
"""

import graphlab
import graphlab.aggregate as agg
import pandas as pd

image_train = graphlab.SFrame('Course1/Week6/image_train_data/')
image_test = graphlab.SFrame('Course1/Week6/image_test_data/')

image_train.head()
image_test.head()

### Qn 1

image_train.groupby('label',operations={'count':agg.COUNT('label')})

### Qn 2 & 3

image_train_bird = image_train[image_train['label'] == 'bird']
image_train_dog= image_train[image_train['label'] == 'dog']
image_train_cat = image_train[image_train['label'] == 'cat']
image_train_auto = image_train[image_train['label'] == 'automobile']
                                    
knn_model_bird = graphlab.nearest_neighbors.create(image_train_bird,features=['deep_features'],label='id')
knn_model_dog = graphlab.nearest_neighbors.create(image_train_dog,features=['deep_features'],label='id')
knn_model_cat = graphlab.nearest_neighbors.create(image_train_cat,features=['deep_features'],label='id')
knn_model_auto = graphlab.nearest_neighbors.create(image_train_auto,features=['deep_features'],label='id')

test_image = image_test[0:1]                  

def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')

cat_neighbors = get_images_from_ids(knn_model_cat.query(test_image))
cat_neighbors['image'].show()

dog_neighbors = get_images_from_ids(knn_model_dog.query(test_image))
dog_neighbors['image'].show()

### Qn 4, 5 & 6

knn_model_cat.query(test_image)['distance'].mean()  ### 36.15573070978294
knn_model_dog.query(test_image)['distance'].mean()  ### 37.77071136184156

### Qn 7

image_test_bird = image_test[image_test['label'] == 'bird']
image_test_dog= image_test[image_test['label'] == 'dog']
image_test_cat = image_test[image_test['label'] == 'cat']
image_test_auto = image_test[image_test['label'] == 'automobile']

dog_cat_neighbors = knn_model_cat.query(image_test_dog, k=1)['distance']
dog_bird_neighbors = knn_model_bird.query(image_test_dog, k=1)['distance']
dog_auto_neighbors = knn_model_auto.query(image_test_dog, k=1)['distance']
dog_dog_neighbors = knn_model_dog.query(image_test_dog, k=1)['distance']

dog_distances = graphlab.SFrame({'dog-automobile': dog_auto_neighbors,'dog-bird': dog_bird_neighbors,'dog-cat': dog_cat_neighbors,'dog-dog': dog_dog_neighbors})
dog_distances.head()

def is_dog_correct(row):
    if (row['dog-dog'] == min(row)):
        return 1
    else:
        return 0

### Predict = 1 if correct else 0
        
dog_distances_df = graphlab.SFrame.to_dataframe(dog_distances)
predict = dog_distances_df.apply(is_dog_correct,axis=1)

sum = float(predict.sum())
accuracy = (sum/len(predict))*100 ### 67.8%
