
#commentfrom divyansh - Vyshnavi you are making the ppt right? also just to confirm m work is train-test-split and regression and if anythin else add it in comment 
 i am doing preprocessing and data collection 
should i share code 
#cmtfromdivyansh - yes upload a file and share the code and dataset 



yes divyansh your work is trin test split the model 
and i am going to do the ppt




# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset

# Step 2: Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("Original Data:\n", df.head())

# Step 3:  missing values for demonstration
df.loc[5:7, 'sepal length (cm)'] = np.nan

# Step 4: Handle missing values (fill with mean)
df['sepal length (cm)'] = df['sepal length (cm)'].fillna(df['sepal length (cm)'].mean())

# Step 5: Feature & target split
X = df.drop("target", axis=1).values
y = df['target'].values

# Step 6: Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 7: Encode labels (not required here since already numeric, but shown for other datasets)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Step 10: Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Print example batch
for batch_X, batch_y in train_loader:
    print("\nSample batch X:", batch_X)
    print("Sample batch y:", batch_y)
    break
    is this ok or should i do other one 

