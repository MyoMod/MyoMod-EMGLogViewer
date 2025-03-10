import re
import matplotlib.pyplot as plt

def decode_array(encoded_str):
    # Remove curly braces and split by comma
    array_str = encoded_str.strip('{}')
    array = [float(num) for num in array_str.split(',')]
    return array

def plot_array(array):
    plt.plot(array)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Array Plot')
    plt.show()

if __name__ == "__main__":
    encoded_str = input("Enter the encoded array string (e.g., {1, 2, 3, 4, 5}): ")
    array = decode_array(encoded_str)
    plot_array(array)
