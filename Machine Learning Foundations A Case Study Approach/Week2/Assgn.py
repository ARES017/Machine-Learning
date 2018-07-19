# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import graphlab
import graphlab.aggregate as agg

sales = graphlab.SFrame('Course1/Week2/home_data.gl/')

sales.head()

### Qn 1
### Zipcode with highest avg house price

df = sales.groupby(key_columns='zipcode', operations={'price': agg.MEAN('price')})

df = df.sort(['price']) 
df.tail() ### zipcode - 98039 & price - 2160606.6

### Qn 2

sales.shape ### 21613 rows & 21 columns

df = sales[(sales['sqft_living'] > 2000) & (sales['sqft_living'] < 4000)]

df.shape ### 9111 rows & 21 columns

fraction = 9111.0/21613.0

### Qn 3

train_data,test_data = sales.random_split(.8,seed=0)

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
my_features_model = graphlab.linear_regression.create(train_data,target='price',features=my_features,validation_set=None)

print my_features_model.evaluate(test_data) ### {'max_error': 3486584.509381705, 'rmse': 179542.4333126903}

advanced_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode', 'waterfront', 'condition', 'grade', 'yr_built', 'yr_renovated']
advanced_features_model = graphlab.linear_regression.create(train_data,target='price',features=advanced_features,validation_set=None)

print advanced_features_model.evaluate(test_data) ### {'max_error': 3706563.3645998696, 'rmse': 162335.43489679348}

rmse_diff = 179542.4333126903 - 162335.43489679348
