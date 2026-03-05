import torch
import numpy as np
import pandas as pd
#Converting python list to tensor
x = torch.tensor([1, 3, 4])
print(x)
print(x.dtype)

#Converting numpy array to tensor
numpy_array = np.array([[1,2,3], [4,5,6]])
torch_tensor_from_numpy = torch.from_numpy(numpy_array)

print(numpy_array)
print(torch_tensor_from_numpy)

#Converting csv data to tensor
df = pd.read_csv('data/data.csv')
print(df)
all_values = df.values
tensor_from_df = torch.tensor(all_values, dtype=torch.float32)
print(tensor_from_df)

#Broadcasting
tensor = torch.tensor([[3.0, 8.0, 1.0],
                       [7.0, 17.0, 2.0],
                       [12.0, 12.0, 1.0]])

vector = torch.tensor([[1.1, 1.0, 5.0]])

broadcasting = tensor * vector
print(broadcasting)