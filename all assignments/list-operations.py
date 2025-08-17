lst = list(map(int, input("50 60: ").split()))
lst = list(map(int, input("50 60: ").split()))
target = int(input("20: "))
lst = list(map(int, input("50 60: ").split()))
lst = list(map(int, input("50 60: ").split()))
lst = list(map(int, input("50 60: ").split()))
largest_even = float("-inf")
largest_odd = float("-inf")
lst = list(map(int, input("5: ").split()))

# Consolidated and cleaned-up list operations script
def find_largest(lst):
    if not lst:
        return None
    return max(lst)

def second_largest(lst):
    unique = list(set(lst))
    if len(unique) < 2:
        return None
    unique.sort(reverse=True)
    return unique[1]

def find_largest_even_odd(lst):
    evens = [x for x in lst if x % 2 == 0]
    odds = [x for x in lst if x % 2 != 0]
    largest_even = max(evens) if evens else None
    largest_odd = max(odds) if odds else None
    return largest_even, largest_odd

def average(lst):
    if not lst:
        return None
    return sum(lst) / len(lst)

def count_occurrences(lst, target):
    return lst.count(target)

def remove_duplicates(lst):
    return list(dict.fromkeys(lst))

def find_odd_occurrence(lst):
    for num in set(lst):
        if lst.count(num) % 2 != 0:
            return num
    return None

def union_lists(list1, list2):
    return list(dict.fromkeys(list1 + list2))

def get_int_list(prompt):
    try:
        return list(map(int, input(prompt).split()))
    except ValueError:
        print("Invalid input. Please enter integers separated by spaces.")
        return []

if __name__ == "__main__":
    lst = get_int_list("Enter numbers separated by space: ")
    print("Largest number:", find_largest(lst))
    print("Second largest number:", second_largest(lst))
    even, odd = find_largest_even_odd(lst)
    print("Largest Even:", even)
    print("Largest Odd:", odd)
    print("Average:", average(lst))
    target = input("Enter a number to count its occurrences: ")
    try:
        target = int(target)
        print(f"{target} occurs {count_occurrences(lst, target)} times")
    except ValueError:
        print("Invalid input for target.")
    print("List after removing duplicates:", remove_duplicates(lst))
    print("Number occurring odd number of times:", find_odd_occurrence(lst))
    # Union of two lists
    list1 = get_int_list("Enter first list for union: ")
    list2 = get_int_list("Enter second list for union: ")
    print("Union of lists:", union_lists(list1, list2))
