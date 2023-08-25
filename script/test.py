import torch
import time
import numpy as np
# a = torch.randn((10000,10000)).to("cuda:0")
# e = torch.zeros((10000,10000)).to("cuda:0")
npa = np.ones((10000,10000))
npa2 = np.zeros((10000,10000))

for i in range(10,84):
    npa2[i][i] = 2 
b = []
# def filter_zero(A):
#     valid_cols = []
#     for col_idx in range(A.size(1)):
#         if not torch.all(A[:, col_idx] == 0):
#             valid_cols.append(col_idx)
#     d1 = A[:, valid_cols]
#     return d1
for i in range(10000):
    c = []
    for j in range(10000):
        c.append(j)
    b.append(c)
tmp = 0

st = time.time()
all = []
for i in range(60):
    alli = []
    for j in range(60):
        allj = []
        for k in range(60):
            allj.append(0)
        for k in range(j+1,60):
            allj[k] = b[5000][5000]
        alli.append(allj)
    all.append(alli)
batch = torch.Tensor(all)
batch = batch + batch.transpose(0,1)
et = time.time()
print(et-st)

st2 = time.time()
def add_dim(input):
    input_shape = input.shape
    new_shape = (1,) + input_shape
    return input.reshape(new_shape)
def concat_arr(input,output):
    if output is None:
        return add_dim(input)
    else:
        return np.concatenate((output,add_dim(input)),axis=0)
all = None
for i in range(60):
    alli = None
    for j in range(60):
        allj = np.zeros(60)
        for k in range(j+1,60):
            allj[k] = b[5000][5000]
        alli = concat_arr(allj,alli)
    all = concat_arr(alli,all)
batch = torch.from_numpy(all)
batch = batch + batch.transpose(0,1)
et2 = time.time()
print(et2-st2)

st = time.time()
new_np = npa*npa2
new_np = new_np[~np.all(new_np == 0, axis=0)]
new_np = new_np[~np.all(new_np == 0, axis=1)]
batch = torch.from_numpy(all)
et = time.time()
print(et-st)

# st2 = time.time_ns()

# f = torch.Tensor(b)
# et2 = time.time_ns()
# print(et2-st2)

# st2 = time.time_ns()
# f = torch.from_numpy(npa)
# et2 = time.time_ns()
# print(et2-st2)

# st2 = time.time_ns()
# g = None
# for i in range(100):
#     c = []
#     for j in range(100):
#         h = []
#         for k in range(100):
#             h.append(1)
#         c.append(h)
#     c = torch.Tensor(c)
#     if g is None:
#         g = c.unsqueeze(0)
#     else:
#         g = torch.cat((g,c.unsqueeze(0)),0)
# et2 = time.time_ns()
# print(et2-st2)

# st2 = time.time_ns()
# d = a*e
# d1 = filter_zero(d)
# d2 = filter_zero(d1.transpose(0,1))
# print(d2.shape)
# print(d2)
# et2 = time.time_ns()
# print(et2-st2)
