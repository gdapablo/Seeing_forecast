import numpy as np, pandas as pd, glob

# -----------------------------------------------
#
# -                 PREPROCESSING
#
# -----------------------------------------------

# --- INT data ---

df1 = pd.read_csv('WHTlocal2020-2021.csv')
df2 = pd.read_csv('WHTlocal2021-2024.csv')
df = pd.concat([df1, df2])

# At this point we have all the data in the proper format. Now we need to split the file
def split_and_save(set, name, num_parts = 4):
    # Splitting the dataset set into n-parts
    split_indices = [i * len(set) // num_parts for i in range(1, num_parts)]
    tt_set_parts = np.split(set, split_indices)

    # Saving each part into a separate file
    for i, part in enumerate(tt_set_parts):
        part.to_csv(f'WHT_data/{name}_set_parts_{i + 1}.csv', index=False)

split_and_save(df, 'WHT_weather', num_parts = 30)

# Training files
files = glob.glob('INT_data/INT_weather_set_parts_*')
files = sorted(files)
INT_weather_parts = [pd.read_csv(file_path) for file_path in files]

# Concatenate the DataFrames into a single training set
df = pd.concat(INT_weather_parts, ignore_index=True)

# Displaying the first 5 rows
print(df.head())

# Check the type of each column
print(df.dtypes)

# We need to convert each category into float32
columns = df.columns

# There are some missing values with '\\N' in the variables. Let's transform into an arbitrary number
for ii in columns[1:]:
    df[ii][df[ii] == '\\N'] = -1
    # Converting into float32
    df[ii] = df[ii].astype('float32')

# Now we can count the number of NaNs in each category
print(df.isna().sum())


# For datetimes we want to split into different columns: year, month, day, hour, minutes, seconds
# First we convert into datetime
df['sampletime'] = pd.to_datetime(df['sampletime'])

# Extract year
df['year'] = df['sampletime'].dt.year

# Extract month
df['month'] = df['sampletime'].dt.month

# Extract day
df['day'] = df['sampletime'].dt.day

# Extract hour
df['hour'] = df['sampletime'].dt.hour

# Extract minute
df['minute'] = df['sampletime'].dt.minute

# Extract second
df['second'] = df['sampletime'].dt.second

# Drop the 'sampletime' column
df.drop(columns=['sampletime'], inplace=True)
