# week 3 assignment 2 pandas
# Question 1
import pandas as pd

# Sample data
data = {'X':[78, 85, 96, 80, 86], 'Y':[84, 94, 89, 83, 86], 'Z':[86, 97, 96, 72, 83]}

# Create DataFrame
df1 = pd.DataFrame(data)

# Display DataFrame
print("✅ Question 1 - DataFrame from Dictionary:")
print(df1)
# Question 2
import numpy as np

# Sample DataFrame
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 
             'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 
                'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# Create DataFrame with labels
df2 = pd.DataFrame(exam_data, index=labels)

print("✅ Question 2 - DataFrame with Index Labels:")
print(df2)

# Question 2.1
print("\n✅ Question 2.1 - DataFrame Summary Info:")
print(df2.info())

# Question 2.2
print("\n✅ Question 2.2 - First 3 Rows:")
print(df2.head(3))

# Question 2.3
print("\n✅ Question 2.3 - 'name' and 'score' Columns:")
print(df2[['name', 'score']])

# Question 2.4
print("\n✅ Question 2.4 - Selected Rows and Columns:")
print(df2.loc[['b', 'd', 'f', 'g'], ['name', 'score']])

# Question 2.5
print("\n✅ Question 2.5 - Attempts > 2:")
print(df2[df2['attempts'] > 2])

# Question 2.6
print("\n✅ Question 2.6 - Number of rows and columns:")
print(f"Rows: {df2.shape[0]}, Columns: {df2.shape[1]}")

# Question 2.7
print("\n✅ Question 2.7 - Score between 15 and 20:")
print(df2[df2['score'].between(15, 20)])

# Question 2.8
print("\n✅ Question 2.8 - Attempts < 2 and Score > 15:")
print(df2[(df2['attempts'] < 2) & (df2['score'] > 15)])

# Question 2.9
print("\n✅ Question 2.9 - Change score in row 'd' to 11.5:")
df2.loc['d', 'score'] = 11.5
print(df2.loc['d'])

# Question 2.10
print("\n✅ Question 2.10 - Mean of all scores:")
mean_score = df2['score'].mean()
print(f"Mean Score: {mean_score}")

# Question 2.11
print("\n✅ Question 2.11 - Append and then delete row 'k':")

# Create a new row 'k'
new_row = pd.DataFrame({
    'name': ['New Student'],
    'score': [13],
    'attempts': [1],
    'qualify': ['yes']
}, index=['k'])

# Append to df2
df2 = pd.concat([df2, new_row])
print("After appending row 'k':")
print(df2.tail())

# Now delete row 'k'
df2 = df2.drop('k')
print("\nAfter deleting row 'k':")
print(df2.tail())

# Question 2.12
print("\n✅ Question 2.12 - Sort by name (desc) and score (asc):")
sorted_df = df2.sort_values(by=['name', 'score'], ascending=[False, True])
print(sorted_df)

# Question 2.13
print("\n✅ Question 2.13 - Replace 'yes' and 'no' with True and False:")
df2['qualify'] = df2['qualify'].map({'yes': True, 'no': False})
print(df2)

# Question 2.14
print("\n✅ Question 2.14 - Change name 'James' to 'Suresh':")
df2['name'] = df2['name'].replace('James', 'Suresh')
print(df2[df2['name'] == 'Suresh'])

# Question 2.15
print("\n✅ Question 2.15 - Delete 'attempts' column:")
df2 = df2.drop(columns='attempts')
print(df2)

# Question 2.16
print("\n✅ Question 2.16 - Write DataFrame to CSV with tab separator:")
df2.to_csv('exam_data_tab.csv', sep='\t', index=True)
print("DataFrame written to 'exam_data_tab.csv' with tab separator.")
