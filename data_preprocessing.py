import pandas as pd


def remove_redundancy(data):
    duplicates = data.duplicated()
    unique_data = data[~duplicates]
    print("no of duplicated rows", duplicates.sum())
    return unique_data


data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Ali', 'Bob'],
    'Age': [25, 30, 28, 25, 32, 30],
    'City': ['New York', 'London', 'Paris', 'New York', 'Tokyo', 'London']
})

clean_data = remove_redundancy(data)
print("\nCleaned Data:")
print(clean_data)
