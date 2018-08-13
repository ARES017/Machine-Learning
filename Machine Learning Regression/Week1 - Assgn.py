### Import required libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

### Read input files

train_data = pd.read_csv('kc_house_train_data.csv')
test_data = pd.read_csv('kc_house_test_data.csv')

### Process input as required

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

for key in dtype_dict:
    train_data[key] = train_data[key].astype(dtype_dict[key])
    test_data[key] = test_data[key].astype(dtype_dict[key])
    
train_data.info()   
test_data.info() 
    
### Creating Linear Regression Model

X_train, X_test, y_train, y_test = train_data['sqft_living'],test_data['sqft_living'], train_data['price'],test_data['price']

model = LinearRegression()
trained_model = model.fit(pd.DataFrame(X_train),pd.DataFrame(y_train))

### Qn1

trained_model.predict(2650)  ### 700074.85

### Qn 2

test_prediction = trained_model.predict(pd.DataFrame(X_test))

residue = test_prediction - pd.DataFrame(y_test)
rss = residue['price'].apply(lambda x:x**2).sum() ### 2.75e+14

### Qn 3

trained_model.coef_ ### Coeff or slope - 281.95883963
trained_model.intercept_ ### Intercept - -47116.07907289

price = 800000
sqft_living = (price - trained_model.intercept_)/trained_model.coef_ ### 3004.39624515

### Qn 4
    
X_train, X_test, y_train, y_test = train_data['bedrooms'],test_data['bedrooms'], train_data['price'],test_data['price']

model = LinearRegression()
trained_model = model.fit(pd.DataFrame(X_train),pd.DataFrame(y_train))

test_prediction = trained_model.predict(pd.DataFrame(X_test))

residue = test_prediction - pd.DataFrame(y_test)
rss = residue['price'].apply(lambda x:x**2).sum() ### 4.93e+14

