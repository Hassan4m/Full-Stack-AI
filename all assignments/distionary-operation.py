# Program to check if a key exists in a dictionary

my_dict = {'name': 'Fatima', 'age': 20, 'city': 'Lahore'}
key = input("Enter key to check: ")

if key in my_dict:
    print(f"'{key}' exists in the dictionary.")
else:
    print(f"'{key}' does not exist in the dictionary.")


# Program to add a key-value pair to the dictionary

my_dict = {'name': 'Fatima', 'age': 20}
key = input("Enter key to add: ")
value = input("Enter value: ")

my_dict[key] = value
print("Updated dictionary:", my_dict)



# Program to find the sum of all values in a dictionary

my_dict = {'a': 10, 'b': 20, 'c': 30}
total = 0

for value in my_dict.values():
    total += value

print("Sum of all items:", total)




# Program to multiply all the values in a dictionary

my_dict = {'a': 2, 'b': 3, 'c': 4}
product = 1

for value in my_dict.values():
    product *= value

print("Product of all items:", product)



# Program to create a dictionary of (x: x*x) for numbers from 1 to n

n = int(input("Enter a number: "))
square_dict = {}

for i in range(1, n+1):
    square_dict[i] = i * i

print("Square dictionary:", square_dict)




# Program to concatenate two dictionaries

dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}

# Method 1: Using update()
dict1.update(dict2)
print("Concatenated dictionary:", dict1)

