# LOAD DATA FROM CSV FILE
import numpy as np
brokered_by, price, acre_lot, city, house_size = np.genfromtxt("E:\codes\Full-Stack-AI\RealEstate-USA.csv", delimiter=',', unpack=True, skip_header = 1, usecols=(0, 2, 5, 7,10)              )

print(brokered_by)
print(price)
print(acre_lot)
print(city)
print(house_size)
# RealEstate-USA price  - statistics operations
print("RealEstate-USA Price mean: " , np.mean(price))
print("RealEstate-USA Price average: " , np.average(price))
print("RealEstate-USA Price std: " , np.std(price))
print("RealEstate-USA Price mod: " , np.median(price))
# operations on house_size
print("RealEstate-USA house_size mean: " , np.mean(house_size))
print("RealEstate-USA house_size average: " , np.average(house_size))
print("RealEstate-USA house_size std: " , np.std(house_size))
print("RealEstate-USA house_size mod: " , np.median(house_size))

# Perform basic arithmetic operations
addition = price + city
subtraction = price - city
multiplication = acre_lot * house_size
division = house_size / city
print(" RealEstate-USA price - city - Addition:", addition)
print(" RealEstate-USA price - city - Subtraction:", subtraction)
print(" RealEstate-USA acre_lot - house_size - Multiplication:", multiplication)
print(" RealEstate-USA house_size - city - Division:", division)

D2 = np.array ([price, house_size])
print("RealEstate-USA Price and House Size Array:\n", D2)

D3 = np.array ([brokered_by, acre_lot, city])
print("RealEstate-USA Brokered By, Acre Lot, City Array:\n", D3)

for item in np.nditer(price):
    print("price Item:", item)
for item in np.nditer(house_size):
    print("house_size Item:", item)
for item in np.ndenumerate(brokered_by) :
    print("brokered_by Item:", item)
for item in np.ndenumerate(acre_lot) :
    print("acre_lot Item:", item)
print("dimension of elements are:",np.ndim(price))
print("shape of elements are:",np.shape(price))
print("size of elements are:",np.size(price))
print("type of elements are:",type(price))

# Splicing array
d2slice = D2[0:2, 0:2]
print("RealEstate-USA price and house_size - 2 dimentional arrary - Splicing", d2slice)
# Splicing array with step
d2slice_step = D2[0:2, 0:4:2]
