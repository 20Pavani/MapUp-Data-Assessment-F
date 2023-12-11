import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    df = df.sort_values(by=['id_1', 'id_2'])
    # Create a pivot table using 'id_1' as index, 'id_2' as columns, and 'car' as values
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Replace NaN values with 0 and set the diagonal values to 0
    car_matrix = car_matrix.fillna(0).values
    for i in range(min(car_matrix.shape[0], car_matrix.shape[1])):
        car_matrix[i, i] = 0

    # Convert the NumPy array back to a DataFrame
    car_matrix = pd.DataFrame(car_matrix, index=df['id_1'].unique(), columns=df['id_2'].unique())
    print("Car Matrix Generation")
    return car_matrix

# Assuming your dataset-1.csv is in the same directory as your script
dataset_path =r"C:\Users\umesh\Downloads\MapUp-Data-Assessment-F-main\MapUp-Data-Assessment-F-main\datasets\dataset-1.csv"

df = pd.read_csv(dataset_path)

result_matrix = generate_car_matrix(df)

print(result_matrix)


def get_type_count(df: pd.DataFrame) -> dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Add a new categorical column 'car_type' based on values of the column 'car'
    df['car_type'] = pd.cut(df['car'], bins=[float('-inf'), 15, 25, float('inf')], labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_counts = dict(sorted(type_counts.items()))

    return type_counts

# Assuming your dataset-1.csv is in the same directory as your script
dataset_path = r"C:\Users\umesh\Downloads\MapUp-Data-Assessment-F-main\MapUp-Data-Assessment-F-main\datasets\dataset-1.csv"
df = pd.read_csv(dataset_path)

result_type_count = get_type_count(df)
print("Car Type Count Calucation")
print(result_type_count)

import pandas as pd

def get_bus_indexes(df: pd.DataFrame) -> list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Calculate the mean of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identify the indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the list of indices in ascending order
    bus_indexes.sort()

    return bus_indexes

# Assuming your dataset-1.csv is in the same directory as your script
dataset_path = r"C:\Users\umesh\Downloads\MapUp-Data-Assessment-F-main\MapUp-Data-Assessment-F-main\datasets\dataset-1.csv"
df = pd.read_csv(dataset_path)

result_bus_indexes = get_bus_indexes(df)
print("Bus Count Index Retrieval")
print(result_bus_indexes)

def filter_routes(df: pd.DataFrame) -> list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Calculate the average 'truck' values for each route
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average 'truck' values are greater than 7
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of route names in alphabetical order
    filtered_routes.sort()

    return filtered_routes

# Assuming your dataset-1.csv is in the same directory as your script
dataset_path = r"MapUp-Data-Assessment-F-main/datasets/dataset-1.csv"
df = pd.read_csv(dataset_path)

result_filtered_routes = filter_routes(df)
print("Route Filtering")
print(result_filtered_routes)

def multiply_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Apply custom conditions to multiply matrix values
    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

# Assuming you have the result_matrix from Question 1
# If not, you can generate it using the generate_car_matrix function
result_matrix = generate_car_matrix(df)

# Apply the multiplication and rounding logic
result_modified_matrix = multiply_matrix(result_matrix)
print("Matrix Value Modification")
print(result_modified_matrix)


import pandas as pd

def time_check(df):
    # Convert the timestamp columns to datetime objects
    df['startTimestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['endTimestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Calculate the time difference for each record
    df['duration'] = df['endTimestamp'] - df['startTimestamp']

    # Group by ('id', 'id_2') and check if the duration covers a full 24-hour period and spans all 7 days
    result = df.groupby(['id', 'id_2']).apply(check_time_range)

    return result
import pandas as pd

def time_check(df):
    # Convert the timestamp columns to datetime objects with a specified format
    df['startTimestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%A %H:%M:%S')
    df['endTimestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%A %H:%M:%S')

    # Calculate the time difference for each record
    df['duration'] = df['endTimestamp'] - df['startTimestamp']

    # Group by ('id', 'id_2') and check if the duration covers a full 24-hour period and spans all 7 days
    result = df.groupby(['id', 'id_2']).apply(check_time_range)

    return result

def check_time_range(group):
    # Check if the duration covers a full 24-hour period
    full_24_hours = group['duration'].min() >= pd.Timedelta(hours=24)

    # Check if the timestamps span all 7 days of the week
    all_days_present = set(group['startTimestamp'].dt.dayofweek.unique()) == set(range(7))

    return pd.Series({'time_check': not (full_24_hours and all_days_present)})

# Load the dataset
df = pd.read_csv(r'C:\Users\umesh\Downloads\MapUp-Data-Assessment-F-main\MapUp-Data-Assessment-F-main\datasets\dataset-2.csv')

# Apply the time_check function to the DataFrame
result_series = time_check(df)

# Print the result
print(" Verify Timestamp Completeness - Output True: Timestamps for each (id, id_2) pair cover a full 24-hour period and span all 7 days of the week.False: Timestamps for at least one (id, id_2) pair are incomplete or incorrect.")
print(result_series)








