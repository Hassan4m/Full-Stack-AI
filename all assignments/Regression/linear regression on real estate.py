import pandas as pd
df = pd.read_csv("E:\codes\Full-Stack-AI\csv files\Real_Estate_Sales_2001-2022_GL-Short (2).csv")

df.columns = df.columns.str.strip()
df.set_index("Serial Number", inplace=True)
print(df)


#Basic DataFrame Methods

print(df.info())

print(df.dtypes)

print(df.describe())

print(df.shape)


#Use to_string() to print with formatting options
print(df.to_string(buf=None, columns=None, col_space=10, header=True, index=True, 
                  na_rep='Missing', formatters=None, float_format=None, sparsify=False, 
                  index_names=True, justify='center', max_rows=10, max_cols=5, 
                  show_dimensions=True, decimal='.', line_width=80))

#7 # Select and print the top 7 rows
print(df.head(7))



# Select and print the bottom 9 rows
print(df.tail(9))



#Access Columns by Name
#'Town' column
print(df['Town'])

# 'Residential Type' column
print(df['Residential Type'])


# Access both 'Town' and 'Date Recorded' columns
print(df[['Town', 'Date Recorded']])



# Access the row where 'Serial Number' is 200008 (loc method)
print(df.loc[200008])


# Access rows where 'Serial Number' is 200305, 200207, 20000048
print(df.loc[[200305, 200207, 20000048]])


# 10. Selecting a slice of rows using .loc
# With index – “Serial Number” value range of “2020090” and “200121” , print the returned row
# and analyze results.
# With index – “Serial Number” value range of “2020090” and “200121” , print the returned row
# With index – “Serial Number” value range of “2020090” and “200121” , print the returned row
print(df.loc[2020090:200121])

# 11. Conditional selection of rows using .loc
# “Sale Amount” greater than “202500”
# , print the returned row and analyze results
print(df.loc[df["Sale Amount"] > 202500])

# 12. Conditional selection of rows using .loc
# “Town” equal to “Anson”
# , print the returned row and analyze results.
print(df.loc[df["Town"] == "Ansonia"])



# 13. Conditional selection of rows using .loc
# “Residential Type” equal to “Condo” and “Assessed Value” is equal to - less than 180500
# , print the returned row and analyze results.
print(df.loc[(df["Residential Type"]=="Condo") & (df["Assessed Value"] < 180500)])


# 14. Selecting a single column using .loc
# With index – “Serial Number” value “201354” , only select following columns
# “Address”,” Assessed Value” , ”Sale Amount” , ”Sales Ratio” , ”Property Type”
# , print the returned row and analyze results.
print(df.loc[201354,['Address','Assessed Value','Sale Amount','Sales Ratio','Property Type']])


# 15. Selecting a slice of columns using .loc
# Form “Date Recorded” to “Sale Amount”
# , print the returned row and analyze results.
print(df.loc[:, 'Date Recorded':'Sale Amount'])

# 16. Combined row and column selection using .loc
# “Residential Type” equal to “Condo” and Columns th “Date Recorded” to “Assessed Value”
# , print the returned row and analyze results.

print(df.loc[df["Residential Type"] == "Condo", "Date Recorded":"Assessed Value"])

# 17. Selecting a single row using .iloc
# Select 5
# th row
# , print the returned row and analyze results
print(df.iloc[5])


# 18. Selecting multiple rows using .iloc
# Select – 7
# th
# row, 9th row and 15th row
# , print the returned row and analyze results.
print(df.iloc[[7,9,15]])


# 19. Selecting a slice of rows using .iloc
# Select from 5th to 13th row
# , print the returned row and analyze results.

rows_5_to_13 = df.iloc[4:13]
print(" i am printing rows_5_to_13",rows_5_to_13)

# 20: Selecting a single column using .iloc
print(df.iloc[:, 2])

#  21: Selecting multiple columns
print(df.iloc[:, [1, 3, 6]])

#  22: Slice of columns (2nd to 5th)
print(df.iloc[:, 1:5])
#  23: Combined row and column selection using .iloc
print(df.iloc[[6, 8, 14], [1, 3]])


#  24: Combined range selection
print(df.iloc[1:6, 1:4])
print("task completed")


#25 add a new row at the end of dataframe
new_data = {
   'Serial Number': 99999999,  
   'List Year': 2025,
   'Date Recorded': '04/22/2025',
    'Town': 'Newtown',
    'Address': '123 AI Street',
    'Assessed Value': 300000,
    'Sale Amount': 500000,
    'Sales Ratio': 0.6,
    'Property Type': 'Residential',
    'Residential Type': 'Single Family',
    'Non Use Code': 32533,
    'Assessor Remarks': 32424,
    'OPM remarks': 214,
    'Location': "Pakistan"
}

df.loc[len(df)] = new_data  
print("\nUpdated DataFrame after adding a new row:")
print(df)


# 26. Delete row with index 2
df = df.drop(df.index[2])

#32: Use query()
df.query("`Assessed Value` < 127400 and `Property Type` == 'Commercial' and `Residential Type` != 'Single Family'")



# 28: Delete column Residential Type
df = df.drop(columns=['Residential Type'])


 #29: Delete columns Assessor Remarks and Location
df = df.drop(columns=['Assessor Remarks', 'Location'])

#question no 30: Rename column List Year to List_Year_Changed
df = df.rename(columns={'List Year': 'List_Year_Changed'})


#31: Rename index label 200400 to 20040333
df = df.rename(index={200400: 20040333})


#33: Sort DataFrame by “Assessed Value” (ascending)
df_sorted = df.sort_values(by='Assessed Value', ascending=True)
print(df_sorted)

#34: Group by “Property Type” and sum “Sale Amount”
grouped = df.groupby('Property Type')['Sale Amount'].sum()
print(grouped)



#35: Use dropna() to remove rows with missing values
df_no_missing = df.dropna()
print(df_no_missing)



#36: Fill all NaN values with 0
df_filled = df.fillna(0)
print(df_filled)
