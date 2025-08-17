#reverse the tuple
my_tuple = (1, 2, 3, 4, 5)
reversed_tuple = ()

for i in range(len(my_tuple) - 1, -1, -1):
    reversed_tuple += (my_tuple[i],)

print("Reversed tuple:", reversed_tuple)

# access the value  20 from the tuple
my_tuple = (10, 20, 30, 40, 50)

for item in my_tuple:
    if item == 20:
        value = item
        break

print("Accessed value:", value)

#swap the tuples
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)

temp = tuple1
tuple1 = tuple2
tuple2 = temp

print("Swapped tuple1:", tuple1)
print("Swapped tuple2:", tuple2)
