from torch.utils.data import DataLoader
import torch
import numpy as np
from model import MyModule
from Dataset import MyDataset
import sklearn.metrics as m
from numba import njit
import os
import warnings
from collate_fn import custom_collate_fn
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

warnings.filterwarnings('ignore')

def RMSE(y_true, y_pred):
    return np.sqrt(m.mean_squared_error(y_true, y_pred))
def MAE(y_true, y_pred):
    return m.mean_absolute_error(y_true, y_pred)
def SD(y,f):
    f = np.array(f)
    y = np.array(y)
    f,y = f.reshape(-1,1),y.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(f,y)
    y_ = lr.predict(f)
    sd = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
    return sd

def R(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

def S(y,f):
    sp = stats.spearmanr(y,f)[0]

    return sp
@njit
def c_index(y_true, y_pred):
    summ = 0
    pair = 0

    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1

    if pair != 0:
        return summ / pair
    else:
        return 0
device = "cuda:2" if torch.cuda.is_available() else "cpu"

path = os.path.abspath(os.path.dirname(os.getcwd()))
data_path = './data'

batch_size = 32
max_seq_len = 1024
max_smi_len = 256

test_dataset = MyDataset('test2020', data_path, max_seq_len, max_smi_len)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

model = MyModule().to(device)

model_dir = './model'
# Open file to save test metrics
with open(os.path.join(model_dir, 'test_metrics.txt'), 'w') as f:
    f.write("Model\tTest_RMSE\tTest_MAE\tTest_SD\tTest_CL\tTest_R\tTest_S\n")

# Model fusion testing module
final_pre = []
target = []
id = []
num_models = 5 

for i in range(num_models):
    model_path = os.path.join(model_dir, f'Experiment_{i + 1}_best_model.ckpt')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    pre = []
    for id_name, smi_encode, seq_encode, affinity in test_dataloader:
        with torch.no_grad():
            final_output = model(seq_encode, smi_encode)
        pre.extend(final_output.cpu().numpy().reshape(-1) / num_models)
        if i == 0:
            id.extend(id_name)
            target.extend(affinity.cpu().numpy().reshape(-1))
    if i == 0:
        final_pre = pre
    else:
        final_pre = [i + j for i, j in zip(final_pre, pre)]

rmse = RMSE(final_pre, target)
mae = MAE(final_pre, target)
sd = SD(final_pre, target)
cl = c_index(final_pre, target)
r = R(final_pre, target)
s = S(final_pre, target)
print(f"模型融合测试集RMSE为{rmse}  测试集MAE为{mae}  测试集SD为{sd}  测试集CL为{cl}  测试集R为{r}  测试集S为{s}")
with open(os.path.join(model_dir, 'test_metrics.txt'), 'a') as f:
    f.write(f"Model Fusion\t{rmse:.4f}\t{mae:.4f}\t{sd:.4f}\t{cl:.4f}\t{r:.4f}\t{s:.4f}\n")
