# Compound Interest Calculator

p = 2500           # Principal amount
r = 7 / 100        # Annual interest rate (7%)
n = 4              # Compounded quarterly
t = 3              # Time in years

A = p * (1 + r / n) ** (n * t)  # Total amount
compound_interest = A - p        # Interest earned

# Display output
print("\nResults:")

print(f"Compound Interest = {compound_interest:.2f}")

print(f"Total Amount (Principal + Interest) = {A:.2f}")
