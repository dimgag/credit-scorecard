# Pandas pre-processing cheatsheet
import pandas as pd

# Load data from a CSV file
df = pd.read_csv('your_data.csv')

# Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Check the dimensions of the DataFrame
print("\nDimensions of the DataFrame:")
print(df.shape)

# Check the data types of each column
print("\nData types of each column:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Drop columns that are not needed
df = df.drop(['column1', 'column2'], axis=1)

# Rename columns
df = df.rename(columns={'old_name': 'new_name'})

# Convert a column to numeric type
df['column_name'] = pd.to_numeric(df['column_name'])

# Convert a column to datetime type
df['date_column'] = pd.to_datetime(df['date_column'])

# Apply a function to a column
df['column_name'] = df['column_name'].apply(function_name)

# Apply a function to each element in a column
df['column_name'] = df['column_name'].map(function_name)

# Create dummy variables for categorical columns
df = pd.get_dummies(df, columns=['categorical_column'])

# Group by a column and calculate mean, sum, etc.
grouped_data = df.groupby('grouping_column')['numeric_column'].mean()

# Sort the DataFrame by one or more columns
df = df.sort_values(by=['column1', 'column2'], ascending=[True, False])

# Reset the index of the DataFrame
df = df.reset_index(drop=True)

# Save the processed data to a new CSV file
df.to_csv('processed_data.csv', index=False)
