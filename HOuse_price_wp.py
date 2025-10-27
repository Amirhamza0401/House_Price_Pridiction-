
"""House price pridiction"""
"""(withput price column)"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

# Ignore all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


pd.set_option("display.max_columns",16)
ds=pd.read_csv(r"C:\Users\Amirhamza\OneDrive\Desktop\House price\house_prices.csv")

"""Preprocessing"""

print(ds.shape)
ds.drop(columns=["Dimensions","Plot Area","Super Area","Society","Status"],inplace=True)
#print(ds.isnull().sum())

ds['Price (in rupees)'] = ds['Price (in rupees)'].fillna(ds['Price (in rupees)'].mean())

categorical_none_cols = ["Car Parking", "overlooking", "facing", "Ownership", "Description"]
for col in categorical_none_cols:
    ds[col] = ds[col].fillna("None")

mode_cols = ['Furnishing', 'Bathroom', 'Transaction']
for col in mode_cols:
    ds[col] = ds[col].fillna(ds[col].mode()[0])
# Replace '> 10' with a number (e.g., 11)
ds['Bathroom'] = ds['Bathroom'].replace('> 10', 11)
# Convert entire column to numeric
ds['Bathroom'] = pd.to_numeric(ds['Bathroom'])


ds['Balcony'] = ds['Balcony'].fillna(0)
ds['Balcony'] = ds['Balcony'].replace('> 10', 11)
# Convert entire column to numeric
ds['Balcony'] = pd.to_numeric(ds['Balcony'])


# Convert to numeric first
ds['Carpet Area'] = ds['Carpet Area'].str.replace(' sqft', '', regex=False)
ds['Carpet Area'] = pd.to_numeric(ds['Carpet Area'], errors='coerce')
ds['Carpet Area'] = ds['Carpet Area'].fillna(ds['Carpet Area'].median())

# Split 'Floor' into floor number and total floors
ds[['Floor_num', 'Total_floors']] = ds['Floor'].str.split(' out of ', expand=True)
ds['Floor_num'] = pd.to_numeric(ds['Floor_num'], errors='coerce').fillna(0)
ds['Total_floors'] = pd.to_numeric(ds['Total_floors'], errors='coerce').fillna(0)
ds.drop(columns=["Floor"],inplace=True)

# Convert amount in float
def to_amount(x):
    if pd.isnull(x):
        return np.nan
    x=str(x).lower().strip()
    
    if 'cr' in x:
        return float(x.replace('cr','').strip())*10000000
    elif 'lac' in x:
        return float(x.replace('lac','').strip())*100000
    else:
        return np.nan
    
ds["Amount(in rupees)"]=ds["Amount(in rupees)"].apply(to_amount)
ds['Amount(in rupees)'] = ds['Amount(in rupees)'].fillna(ds['Amount(in rupees)'].median())

print(ds["Amount(in rupees)"].dtype)
l=ds["location"].value_counts()
ds["Location_freq"]=ds["location"].map(l)

# Get frequency counts
facing_counts = ds['facing'].value_counts()
ds['facing_freq'] = ds['facing'].map(facing_counts)

car_counts = ds['Car Parking'].value_counts()
ds['Car_Parking_freq'] = ds['Car Parking'].map(car_counts)

location_counts = ds["location"].value_counts()
facing_counts = ds["facing"].value_counts()
car_counts = ds["Car Parking"].value_counts()

ds.drop(columns=['location','facing','Car Parking' ], inplace=True)

"""Drop Price Column"""
ds.drop(columns=['Price (in rupees)'], inplace=True)

"""LAbel Encoding"""
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
ds['Transaction_encoded'] = le.fit_transform(ds['Transaction'])
ds['Furnishing_encoded'] = le.fit_transform(ds['Furnishing'])
ds['Ownership_encoded'] = le.fit_transform(ds['Ownership'])

ds.drop(columns=['Transaction','Ownership'], inplace=True)
ds.drop(columns=['Furnishing','Description'], inplace=True)

"""Handaling overlooking column"""
from sklearn.preprocessing import MultiLabelBinarizer
ds['Overlooking_list'] = ds['overlooking'].str.split(',\s*')
mlb = MultiLabelBinarizer()
mlb1 = mlb.fit_transform(ds['Overlooking_list'])
overlook_df = pd.DataFrame(mlb1, columns=mlb.classes_)

# Concatenate with original dataset
ds = pd.concat([ds, overlook_df], axis=1)
ds.drop(columns=['overlooking', 'Overlooking_list'], inplace=True)

ds['BHK'] = ds['Title'].str.extract(r'(\d+)\s*BHK', expand=False)
ds['BHK'] = ds['BHK'].astype(float) 
ds.drop(columns=['Title'], inplace=True)

print(ds.info())

bhk_mode = ds['BHK'].mode()[0]  
ds['BHK'].fillna(bhk_mode, inplace=True)

x=ds.drop(columns=["Amount(in rupees)","Index"])
y=ds["Amount(in rupees)"]
sns.pairplot(ds[['Carpet Area', 'Bathroom', 'Balcony', 'BHK', 'Amount(in rupees)']].sample(500, random_state=42))
plt.show()

x.select_dtypes(include=['float64', 'int64']).boxplot(figsize=(15,6))
plt.xticks(rotation=90)
plt.show()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x1=sc.fit_transform(x)
x=pd.DataFrame(x1,columns=x.columns)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,         # number of trees
    max_depth=10,  
    random_state=42,
)

rf.fit(x_train, y_train)
actual_amount = ds["Amount(in rupees)"].copy()

y1=rf.predict(x)
ds["Actual Amount"] = actual_amount
ds["Predicted Amount"]=y1
pd.set_option('display.float_format', '{:,.2f}'.format)

print(ds[["Actual Amount", "Predicted Amount"]].head(10))

plt.figure(figsize=(8,6))
plt.plot(y.values[:50], label="Actual Price", color='blue', marker='o')
plt.plot(y1[:50], label="Predicted Price", color='red', marker='x')
plt.xlabel("Actual Price (in rupees)")
plt.ylabel("Predicted Price (in rupees)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()

new_data = {}

for col in x.columns:
    val = input(f"{col}: ")
    
    # Handle frequency columns
    if col == "Location_freq":
        if val in location_counts.index:
            val = location_counts[val]
        else:
            val = 1  # default if location not found
    elif col == "facing_freq":
        if val in facing_counts.index:
            val = facing_counts[val]
        else:
            val = 1
    elif col == "Car_Parking_freq":
        if val in car_counts.index:
            val = car_counts[val]
        else:
            val = 1
    
    new_data[col] = [val]

new_data = pd.DataFrame(new_data)
new_data_transform = sc.transform(new_data)
new_prediction = rf.predict(new_data_transform)
print("Predicted Amount (in rupees):", new_prediction)
