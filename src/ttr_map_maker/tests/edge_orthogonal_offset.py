"""
author: GPT-4 with Wolfram Plugin (16.06.2023)
"""
def calculate_offset_1(i, n, d=1):
    # i and n start at 1
    return (2*i - n - 1) * d / 2

def calculate_offset_0(i, n, d=1):
    # i and n start at 0
    return (i - n/2) * d

def test_offset_function(func, start_index, end_index):
    print(f"\nTesting function: {func.__name__}")
    for n in range(start_index, end_index + 1):
        offsets = [func(i, n) for i in range(start_index, n + 1)]
        print(f"n = {n}: {offsets}")

# Test the functions
test_offset_function(calculate_offset_1, 1, 7)
test_offset_function(calculate_offset_0, 0, 6)
