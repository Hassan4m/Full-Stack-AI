basic = 90000

#calcualte HRA and DA
hra = 0.20 * basic
da = 0.15 * basic

#calculate basic salary
total = basic + hra + da    

# results
print(f"\nHRA (20%) = {hra}")
print(f"n DA (15%) = {da}")
print(f"total salary = {total}")