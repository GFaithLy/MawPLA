from torch.utils.data import DataLoader,random_split
import torch
import numpy as np
from torch import nn,optim
from model_old import MyModule
from Dataset import MyDataset
import sklearn.metrics as m
from tqdm import tqdm
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from io import BytesIO
from IPython.display import Image
import numpy as np
import torch
from torch_geometric.utils import to_undirected
from collate_fn import custom_collate_fn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr
device = "cuda:2" if torch.cuda.is_available() else "cpu"
data_cache = {}
def RMSE(y_true, y_pred):
    return np.sqrt(m.mean_squared_error(y_true, y_pred))

def MAE(y_true, y_pred):
    return m.mean_absolute_error(y_true, y_pred)

def pearson(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def spearman(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]

batch_size = 32
epochs = 50
path = os.path.abspath(os.path.dirname(os.getcwd()))
data_path = '/MawPLA/data'
model_dir = './model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

max_seq_len = 1024
max_smi_len = 256

seed = 990721

# GPU uses cudnn as backend to ensure repeatable by setting the following (in turn, use advances function to speed up training)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed)
np.random.seed(seed)

# Load training dataset
train_dataset = MyDataset('train2020', data_path, max_seq_len, max_smi_len)


metric_file_path = os.path.join(model_dir, "metrics.txt")
with open(metric_file_path, 'w') as f:
    f.write("Experiment\tEpoch\tTrain_RMSE\tTrain_MAE\tTrain_Pearson\tTrain_Spearman\tTest_RMSE\tTest_MAE\tTest_Pearson\tTest_Spearman\n")



num_experiments = 5
for experiment in range(num_experiments):
    print(f"Experiment {experiment + 1}/{num_experiments}")

    # train:val=9:1
    train_subset, val_subset = random_split(train_dataset, [int(len(train_dataset) * 0.9),
                                                           len(train_dataset) - int(len(train_dataset) * 0.9)],
                                            generator=torch.Generator().manual_seed(seed + experiment))

    # Create a data loader for training and validation datasets
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Reinitialize the model before each experiment begins
    model = MyModule().to(device)

    # use the corresponding data loader during training and validation
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
    criterion = nn.MSELoss()

    train_rmse_list = []
    train_mae_list = []
    train_pearson_list = []
    train_spearman_list = []
    val_rmse_list = []
    val_mae_list = []
    val_pearson_list = []
    val_spearman_list = []

 
    patience = 5
    no_improvement_count = 0
    best_val_rmse = float('inf')

    for epoch in range(epochs):
        model.train()
        train_pre = []
        train_target = []
        for id_name, smi_encode, seq_encode, affinity in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            final_output = model(seq_encode, smi_encode)
            loss = criterion(final_output, affinity)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_pre.extend(final_output.cpu().detach().numpy().reshape(-1))
            train_target.extend(affinity.cpu().detach().numpy().reshape(-1))

        train_rmse = RMSE(train_target, train_pre)
        train_mae = MAE(train_target, train_pre)
        train_pearson = pearson(train_target, train_pre)
        train_spearman = spearman(train_target, train_pre)

        train_rmse_list.append(train_rmse)
        train_mae_list.append(train_mae)
        train_pearson_list.append(train_pearson)
        train_spearman_list.append(train_spearman)

        # Calculate the validation loss
        model.eval()
        val_pre = []
        val_target = []
        for id_name, smi_encode, seq_encode, affinity in val_dataloader:
            with torch.no_grad():
                final_output = model(seq_encode, smi_encode)
            val_pre.extend(final_output.cpu().detach().numpy().reshape(-1))
            val_target.extend(affinity.cpu().detach().numpy().reshape(-1))

        val_rmse = RMSE(val_target, val_pre)
        val_mae = MAE(val_target, val_pre)
        val_pearson = pearson(val_target, val_pre)
        val_spearman = spearman(val_target, val_pre)

        val_rmse_list.append(val_rmse)
        val_mae_list.append(val_mae)
        val_pearson_list.append(val_pearson)
        val_spearman_list.append(val_spearman)

        with open(metric_file_path, 'a') as f:
            f.write(f"{experiment + 1}\t{epoch + 1}\t{train_rmse:.4f}\t{train_mae:.4f}\t{train_pearson:.4f}\t{train_spearman:.4f}\t{val_rmse:.4f}\t{val_mae:.4f}\t{val_pearson:.4f}\t{val_spearman:.4f}\n")


        print(f'Experiment {experiment + 1} - Epoch {epoch + 1} - Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}, '
              f'Train Pearson: {train_pearson:.4f}, Train Spearman: {train_spearman:.4f}, Test RMSE: {val_rmse:.4f}, '
              f'Test MAE: {val_mae:.4f}, Test Pearson: {val_pearson:.4f}, Test Spearman: {val_spearman:.4f}')

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    #clear
    data_cache.clear()
    # save
    torch.save(model.state_dict(), os.path.join(model_dir, f'Experiment_{experiment + 1}_best_model.ckpt'))

    
    
