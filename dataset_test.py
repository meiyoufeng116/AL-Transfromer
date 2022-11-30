import os

from tqdm import trange
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from matplotlib.transforms import Transform
from data.data_process import dataset_split, dataset_split_h
import numpy as np
import torch
from torch import batch_norm, nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import numpy as np
from torchvision import transforms
from sklearn import preprocessing
import numpy as np  #导入numpy包，用于生成数组
import seaborn as sns  #习惯上简写成sns
sns.set()           #切换到seaborn的默认运行配置


dataset, mask = torch.load('./datasets/normalized_dataset_new.pt')
# norm=torch.nn.InstanceNorm1d(55)
# dataset=norm(dataset.permute(0,2,1))
# dataset=dataset.permute(0,2,1)
train_data, train_mask, valid_data, valid_mask, test_data, test_mask = dataset_split_h(
    dataset, mask)  # b t input
print(np.mean(test_data.detach().numpy()))
print(np.var(test_data.detach().numpy()))

print(torch.nonzero(test_mask).shape)
miss=[]
position=[]

for i in range(test_data.shape[0]):
    for j in range(test_data.shape[1]):
        if test_data[i,j,:].equal(torch.zeros(49)):
            miss.append(i)
            position.append(j)
            break

print(len(miss))
plt.title("Early patient discharge in Test set Total: "+str(len(miss)))
plt.xlabel("Time step")
plt.ylabel("Num of patients")
plt.hist(position)
plt.show()
# norm1=torch.nn.InstanceNorm1d(49)
# train_data=norm1(train_data.permute(0,2,1))
# train_data=train_data.permute(0,2,1)

# print(np.mean(train_data.detach().numpy()))
# print(np.var(train_data.detach().numpy()))
# dataset=(dataset-np.mean(dataset.numpy()))/np.var(dataset.numpy())
# print(np.mean(dataset.numpy()))
# print(np.var(dataset.numpy()))
# train_data=transforms.ToTensor()(train_data.numpy())
# Train=transforms.Normalize(train_data)
# valid_data=transforms.Normalize(valid_data)

