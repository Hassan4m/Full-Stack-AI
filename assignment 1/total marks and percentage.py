#total marks each subject 100
# marks for 5 subjects
sub1 = 70
sub2 = 65
sub3 = 75
sub4 = 80 
sub5 = 40

# Calculate total, average, and percentage
total = sub1 + sub2 + sub3 + sub4 + sub5
average = total / 5
percentage = (total / 500) * 100 

# results
print(f"\ntotal marks = {total}")
print(f"average marks = {average:.2f}")
print(f"percentage = {percentage:.2f}%")