import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('FSL_Data/ham10000/HAM10000_metadata.csv')

# Split the data
train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Verify the splits
print(f"Training data: {len(train_data)} samples")
print(f"Validation data: {len(val_data)} samples")
print(f"Testing data: {len(test_data)} samples")

# Save the splits
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)