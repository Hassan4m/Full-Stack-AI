input_str = input("M.hassan")

if len(input_str) == 0:
    print("Input string is empty")
else:
    first_char = input_str[0]
    last_char = input_str[-1]
    middle_char = input_str[len(input_str) // 2] if len(input_str) > 1 else ""

    new_str = first_char + middle_char + last_char
    print("New string:", new_str)


# write a program to count occurrences of all characters in the input string.
input_str = input("Enter a string to get all occurrences of characters: ")
char_count = {}

for char in input_str:
    char_count[char] = char_count.get(char, 0) + 1

print("Character occurrences:")
for char, count in char_count.items():
    print(f"'{char}': {count}")

 # reverse a string
input_str = input("Shahzad")
reversed_str = ""

for char in input_str:
    reversed_str = char + reversed_str

print("Reversed string:", reversed_str)

# split a string on hyphens
input_str = input("Ai: ")
result = []
temp = ""

for char in input_str:
    if char == "-":
        result.append(temp)
        temp = ""
    else:
        temp += char

result.append(temp)

print("Split parts:", result)

#remove special symbols/punctuation from a string
input_str = input("abdullah ")
cleaned_str = ""

for char in input_str:
    if char.isalnum() or char.isspace():
        cleaned_str += char

print("String without special symbols/punctuation:", cleaned_str)

