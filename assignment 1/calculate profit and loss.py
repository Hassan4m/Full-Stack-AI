cost_price = 500
selling_price = 1200
if selling_price > cost_price:
    profit = selling_price - cost_price
    print(f"profit = {profit:.2f}")
elif cost_price > selling_price:
    loss = cost_price - selling_price
    print(f"loss = {loss:.2f}")
else:
    print("no proft, no loss.")