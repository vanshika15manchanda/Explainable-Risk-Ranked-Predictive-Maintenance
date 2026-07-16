print('--- STEP 2: DATA CLEANING ---')

print('\nChecking for missing values:')
display(df.isnull().sum().to_frame(name='Missing Values'))

# Dropping irrelevant columns (UID and Product ID)
print('\nDropping UDI and Product ID columns...')
df = df.drop(columns=['UDI', 'Product ID'])
print('Columns after dropping:', df.columns.tolist())

print('\nVerifying data types after cleaning:')
df.info()
