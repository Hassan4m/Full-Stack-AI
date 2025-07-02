def celsius_to_fahrenheit(celsius):
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit
celsius = float(input(40))
fahrenheit = celsius_to_fahrenheit(celsius)
print(f"{celsius}\u00B0 C is equal to {fahrenheit}\u00B0 F")