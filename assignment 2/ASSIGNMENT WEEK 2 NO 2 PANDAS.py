# ASSIGNMENT NO 2 PANDAS
import pandas as pd

# Load the CSV file from GitHub
url = "E:\codes\Full-Stack-AI\RealEstate-USA.csv"
df = pd.read_csv(url)

# Print the entire DataFrame
print(df)
# Info about DataFrame
print("\nDataFrame Info:")
print(df.info())

# Data types of each column
print("\nData Types:")
print(df.dtypes)

# Statistical summary
print("\nStatistical Description:")
print(df.describe(include='all'))  # include='all' shows categorical too

# Shape of DataFrame
print("\nShape (rows, columns):")
print(df.shape)
# Default to_string (returns full DataFrame as string)
print("\nDefault .to_string():")
print(df.to_string())

# Showing only specific columns
print("\nShowing selected columns only:")
print(df.to_string(columns=['street', 'city', 'state', 'price']))


# Custom column spacing
print("\nWith col_space=20:")
print(df.to_string(col_space=20))

# Hide index
print("\nWithout index:")
print(df.to_string(index=False))

# Replace NaN with custom value
print("\nReplace NaN with 'Missing':")
print(df.to_string(na_rep='Missing'))

# Format float values
print("\nFormatted floats (2 decimal places):")
print(df.to_string(float_format='{:0.2f}'.format))

# Justify column headers
print("\nJustified headers:")
print(df.to_string(justify='center'))

# Limit number of rows
print("\nMax rows = 5:")
print(df.to_string(max_rows=5))

# Show dimensions
print("\nShow dimensions:")
print(df.to_string(show_dimensions=True))

# Decimal with comma
print("\nDecimal with comma:")
print(df.to_string(decimal=','))
print("\nTop 7 rows of the DataFrame:")
print(df.head(7))

# Bottom 9 Rows
print("\nBottom 9 rows:")
print(df.tail(9))

# Access Single Columns – “city” and “street”
print("\nCity Column:")
print(df['city'])

# Access Multiple Columns – “street” and “city”
print("\nStreet Column:")
print(df['street'])
print("\nStreet and City Columns:")
print(df[['street', 'city']])

# Select a Single Row using
print("\nRow with index 5:")
print(df.loc[5])

# Select Multiple Rows using
print("\nRows with index 3, 5, 7:")
print(df.loc[[3, 5, 7]])

# Select a Range of Rows using:
print("\nRows from index 3 to 9 (inclusive):")
print(df.loc[3:9])

#  Conditional Row Selection
print("\nRows with price > 100000:")
print(df.loc[df['price'] > 100000])

# Conditional Row Selection
print("\nRows where city is 'Adjuntas':")
print(df.loc[df['city'] == 'Adjuntas'])
print("\nRows where city is 'Adjuntas' and price < 180500:")
print(df.loc[(df['city'] == 'Adjuntas') & (df['price'] < 180500)])

#  Select Specific Columns for Index 7
print("\nSelected columns at index 7:")
print(df.loc[7, ['city', 'price', 'street', 'zip_code', 'acre_lot']])

# Select a Slice of Columns
print("\nSlice of columns from 'city' to 'zip_code':")
print(df.loc[:, 'city':'zip_code'])  # selects all rows, sliced columns

# Combined Row + Column Selection 
print("\nRows where city is 'Adjuntas' and columns from 'city' to 'zip_code':")
print(df.loc[df['city'] == 'Adjuntas', 'city':'zip_code'])

# Select the 5th Row using 
print("\n17. 5th row (index 4):")
print(df.iloc[4])

# Select 7th, 9th, and 15th Rows
print("\n18. 7th, 9th, and 15th rows:")
print(df.iloc[[6, 8, 14]])

# Select Rows from 5th to 13th
print("\n19. Rows from 5th to 13th (indexes 4 to 12):")
print(df.iloc[4:13])

# Select the 3rd Column
print("\n20. 3rd column (index 2):")
print(df.iloc[:, 2])  # This selects all rows of the 3rd column

# Select 2nd, 4th, and 7th Columns
print("\n21. 2nd, 4th, and 7th columns (indexes 1, 3, 6):")
print(df.iloc[:, [1, 3, 6]])

# Slice Columns from 2nd to 5th
print("\n22. Columns from 2nd to 5th (indexes 1 to 4):")
print(df.iloc[:, 1:5])

# Combine Row + Column Selection 
print("\n23. Rows 7, 9, 15 and columns 2nd & 4th (indexes 6, 8, 14 and 1, 3):")
print(df.iloc[[6, 8, 14], [1, 3]])

# Combine Range of Rows (2, 6) and Columns (2nd to 4th)
print("\n24. Rows 2 to 6 (indexes 1 to 5) and columns 2nd to 4th (indexes 1 to 3):")
print(df.iloc[1:6, 1:4])

# Add a New Row to the DataFrame
print("\n25. Adding a new row:")
new_row = {
    'brokered_by': 'Test Agency',
    'status': 'for_sale',
    'price': 99999,
    'bed': 3,
    'bath': 2,
    'acre_lot': 0.5,
    'street': '123 Test Street',
    'city': 'Testville',
    'state': 'TS',
    'zip_code': 12345,
    'house_size': 1500,
    'prev_sold_date': '2020-01-01'
}

# Append new row and reset index

import pandas as pd

new_row = {
    'brokered_by': 'Test Agency',
    'status': 'for_sale',
    'price': 99999,
    'bed': 3,
    'bath': 2,
    'acre_lot': 0.5,
    'street': '123 Test Street',
    'city': 'Testville',
    'state': 'TS',
    'zip_code': 12345,
    'house_size': 1500,
    'prev_sold_date': '2020-01-01'
}

# Convert the new row into a DataFrame
new_row_df = pd.DataFrame([new_row]) 
df = pd.concat([df, new_row_df], ignore_index=True)

# Print the newly added row
print("\n✅ New row added successfully:")
print(df.tail(1))


# Delete Row with Index 2
print("\n26. Delete row with index 2:")
df = df.drop(index=2)
print(df.head(5))  # Confirm row 2 is gone

#  Delete Rows from Index 4 to 7
print("\n27. Delete rows from index 4 to 7 (inclusive):")
df = df.drop(index=range(4, 8))
print(df.head(10))  # Confirm deletion

# Delete house_size column
print("\n✅ Question 28 - Delete 'house_size' column:")
df = df.drop(columns='house_size')
print(df.head())

# Delete house_size and state columns
print("\n✅ Question 29 - Delete 'house_size' and 'state' columns:")
df = df.drop(columns=['state'], errors='ignore')  # ignore if already removed
print(df.head())

# Reload the data to show renaming
df2 = pd.read_csv("E:\codes\Full-Stack-AI\RealEstate-USA.csv")
df2.rename(columns={'state': 'state_Changed'}, inplace=True)
print("\n✅ Question 30 - Renamed 'state' to 'state_Changed':")
print(df2[['city', 'state_Changed']].head())

# Rename label (row index) from 3 to 5
print("\n✅ Question 31 - Rename index 3 to 5:")
df.rename(index={3: 5}, inplace=True)
print(df.head(10))

# Filter using .query() — price < 127400 and city != 'Adjuntas'
print("\n✅ Question 32 - Filter using .query() where price < 127400 and city != 'Adjuntas':")
filtered_df = df.query('price < 127400 and city != "Adjuntas"')
print(filtered_df)

# Sort DataFrame by price in ascending order
print("\n✅ Question 33 - Sort by price ascending:")
df_sorted = df.sort_values(by='price', ascending=True)
print(df_sorted[['city', 'price']].head())

# Group by city and sum price
print("\n✅ Question 34 - Group by city and sum of price:")
grouped_df = df.groupby('city')['price'].sum().reset_index()
print(grouped_df.head())

# Use dropna() to remove rows with any missing values
print("\n✅ Question 35 - Drop rows with any NaN values:")
df_clean = df.dropna()
print(df_clean)

# Fill all NaN values with
print("\n✅ Question 36 - Fill NaN with 0:")
df_filled = df.fillna(0)
print(df_filled)
