# import libraries
import pandas as pd
import numpy as n
from sqlalchemy import create_engine


# load messages dataset
messages = pd.read_csv('messages.csv')
messages.head()

# load categories dataset
categories = pd.read_csv('categories.csv')
categories.head()

# merge datasets
df = pd.merge(messages, categories, on = 'id')
df.head()

# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';', expand=True)
categories.head()

# select the first row of the categories dataframe
row = categories.iloc[1]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
category_colnames = row.apply(lambda x: x.split('-')[0])
print(category_colnames)

# rename the columns of `categories`
categories.columns = category_colnames
categories.head()

for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
categories.head()

# drop the original categories column from `df`
df.drop('categories', axis = 1, inplace = True)

df.head()

# concatenate the original dataframe with the new `categories` dataframe
df = df.join(categories)
df.head()

# check number of duplicates
sum(df.duplicated())

# drop duplicates
df = df.drop_duplicates()

# check number of duplicates
sum(df.duplicated())

#Save the clean dataset into an sqlite database.
engine = create_engine('sqlite:///DisasterResponse.db')
df.to_sql('DisasterResponse', engine, index=False)