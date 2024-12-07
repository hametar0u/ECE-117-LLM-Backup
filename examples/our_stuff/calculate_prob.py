import math

num_params = 130 * 10**6

for p in [1e-9, 1e-8]:

    p_fliptonan = p * (1 - p)**7 * (1 - 2**-23)

    print(f"ber: {p}")
    print(f"p_fliptonan: {p_fliptonan}")
    print(f"overall: {1 - (1 - p_fliptonan) ** num_params}")
    print("===")