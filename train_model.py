import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("c:\\Users\\bhawn\\Downloads\\Housing.csv")

# Keep Only Needed Columns
df = df[['area', 'mainroad', 'furnishingstatus', 'basement', 'price']]

# Convert Text to Numbers
df['mainroad'] = df['mainroad'].replace({
    'yes': 1,
    'no': 0
})

df['basement'] = df['basement'].replace({
    'yes': 1,
    'no': 0
})

df['furnishingstatus'] = df['furnishingstatus'].replace({
    'furnished': 0,
    'semi-furnished': 1,
    'unfurnished': 2
})

# Features
X = df[['area', 'mainroad', 'furnishingstatus', 'basement']]

# Target
y = df['price']

# Train Model
model = LinearRegression()

model.fit(X, y)

# Save Model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("MODEL TRAINED SUCCESSFULLY")