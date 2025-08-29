#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pathlib import Path
import models.End_to_End_model as End_to_End_model
import matplotlib as mpl  
import os 
import time
import utils.utils_function as utils_function

mpl.rcParams.update(mpl.rcParamsDefault)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
#%% functions
# Define loss function 
def sum_loss(wa, wd, h, U, batch_size, noise_var_DL):
    # Ensure complex conjugate transpositions
    a1 = torch.conj(torch.transpose(wa, -2, -1)) @ torch.conj(torch.transpose(h, -2, -1))
    a2 = torch.conj(torch.transpose(wd, -2, -1)) @ a1
    a3 = h @ wa @ wd @ a2
    
    # Construct matrix g = I + (H * Wa * Wd * Wd^H * Wa^H * H^H) / (n * sigma^2)
    g = torch.eye(U, device=h.device, dtype=h.dtype).reshape((1, U, U)) + (a3 / (U * noise_var_DL))
    # return g
    # Use numerically stable log-determinant computation
    s = torch.log(torch.det(g))  # Log-determinant
    loss = torch.sum(torch.abs(s)) / batch_size

    return -loss  # Negative for optimization

# Define evaluation function 
def evaluate_sum_rate(h, WA, WD, U, noise_var_DL, batch_size, num_iter):
    sum_rate = torch.zeros(num_iter+1)
    for i in range(num_iter+1):
        sum_rate[i] = sum_loss(WA[i], WD[i], h, U, batch_size, noise_var_DL)
       
        
    return -sum_rate

#ploting the results
def plot_sum_rate(sum_rate,pga_type,channel_type,sum_rate_0=None):
    y = sum_rate.cpu().detach().numpy()
    x = np.arange(len(y)) 

    plt.figure()
    plt.plot(x, y, '+-',label='after training')
    if sum_rate_0 is not None:
        y0=sum_rate_0.cpu().detach().numpy()
        plt.plot(x, y0, '+--',color='grey',label='before training')
        
    plt.title(f'Sum rate, {pga_type}, {channel_type} A={A} U={U} L={L} T={T}')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Achievable Rate')
    plt.grid()
    plt.legend()
    plt.xlim(left=0)
    plt.show()

def save_sum_rate(file_name,sum_rate_):
    # save sum rate
    sum_rate={}
    sum_rate[f'{file_name}']=sum_rate_.detach().numpy()
    save_dir=path_init/f'sumRate/{noise_var_DL:.0e}/L_{L}_T_{T}'
    os.makedirs(save_dir,exist_ok=True)
    np.savez(save_dir/f'{file_name}', **sum_rate)

#%% Parameters to modify for each method (args)
noise_var = 2e-3 # Uplink noise variance
U     = 4        # Num of users
A     = 64       # BS antennas

# noise_var_DL=A*noise_var # downlink noise variance
noise_var_DL=noise_var # downlink noise variance
mpnet_Supervised=True
Constrained=False
Constrained_notConstrained="mpnet_c" if Constrained else "mpnet"
mpnet_sup_unsup="sup" if mpnet_Supervised else "unsup"
estimator=f'{Constrained_notConstrained}_{mpnet_sup_unsup}'
L     = 16       # RF chains
T     = 1       # Instant
print(f"Uplink noise vrariance={noise_var:.0e}, Downlink noise variance={noise_var_DL:.0e}, L={L}, T={T}, supervised={mpnet_Supervised}")
# %% Parameters defining
k=5 # iteration mpnet sc
num_of_iter_pga_unf=10 # iteration pga


max_batch=300


path_init=Path.cwd()/'.saved_data'
save_dir=path_init/f'pretrained_E2E_models/{noise_var_DL:.0e}'
os.makedirs(save_dir,exist_ok=True)
#%%----------------------------------- Load Data --------------------------------------------------------
# load test data 

dataset_dir = f'Data/channels_var_snr/{noise_var:.0e}/T_{T}'  
test_data = np.load(path_init/dataset_dir/'test_data.npz')
# Get Measurement matrix
M_data=np.load(path_init/'Data'/f'Measurement_matrix/L_{L}_T_{T}'/'test.npz')  

h_test        =    torch.tensor(test_data['h'],dtype=torch.complex128)       
h_noisy_test  =    torch.tensor(test_data['h_noisy'],dtype=torch.complex128)      
sigma_2_test  =    torch.tensor(test_data['sigma_2'])     
M_test = torch.tensor(M_data['M_test'],dtype=torch.complex128)
M_tilde_test = torch.tensor(M_data['M_tilde_test'],dtype=torch.complex128)
# the observed signal
y_test = torch.matmul((h_noisy_test).unsqueeze(1),torch.conj(M_tilde_test)).squeeze().to(device)

## preprocessing 
norm_y_test = torch.norm(y_test, p=2, dim=1)
y_test = y_test / norm_y_test[:, None]
# normalize channels 
h_noisy_test   = h_noisy_test / (norm_y_test[:,None]/np.sqrt(L))
h_test   = h_test / (norm_y_test[:,None]/np.sqrt(T*L))
sigma_2_test = torch.sqrt(sigma_2_test) / norm_y_test


#load train data
start_time = time.time()

H = []
H_noisy = []
M = []
M_tilde = []

batch_idx=0
while batch_idx < max_batch:

    train_data = np.load(path_init/dataset_dir/f'batch_{batch_idx}.npz')
    m_data=np.load(path_init/'Data'/f'Measurement_matrix/L_{L}_T_{T}'/f'batch_{batch_idx}.npz')
   
    # true channels 
    h = torch.tensor(train_data['h'], dtype=torch.complex128)
    h_noisy = torch.tensor(train_data['h_noisy'], dtype=torch.complex128)
    #measurement matrices
    m = torch.tensor(m_data['M_train'],dtype=torch.complex128)
    m_tilde = torch.tensor(m_data['M_tilde_train'],dtype=torch.complex128)
    #expand m matrices
    m=m.expand(h_noisy.shape[0], -1, -1)
    m_tilde=m_tilde.expand(h_noisy.shape[0], -1, -1)
    
    # Append to lists
    H.append(h)
    H_noisy.append(h_noisy)
    M.append(m)
    M_tilde.append(m_tilde)

    batch_idx+=1

H = torch.cat(H, dim=0)
H_noisy = torch.cat(H_noisy, dim=0)
M = torch.cat(M, dim=0)
M_tilde = torch.cat(M_tilde, dim=0)
#the observed signal:
Y=torch.matmul((H_noisy).unsqueeze(1),torch.conj(M_tilde)).squeeze()
#noise variance vector:
sigma_2=torch.sqrt(torch.tensor([noise_var] * len(H)))

#----------NORMALIZATION------------------------
norm_Y     =    torch.norm(Y,p=2,dim=1)
Y  = Y/ norm_Y[:,None]
H=H/(norm_Y[:,None]/np.sqrt(T*L))
H_noisy=H_noisy/(norm_Y[:,None]/np.sqrt(L))
sigma_2=sigma_2/ norm_Y

end_time = time.time()
print(f"Cell execution time: {end_time - start_time:.4f} seconds")

#################################################################################################
##%%-------------------------------Get train, validation and test data -------------------------------------------------

train_ratio=0.8
split_index = int(len(H) * train_ratio)

# train data 
H_train       = H [:split_index].to(device)
H_noisy_train = H_noisy[:split_index] # noisy channels normalizd
Y_train       = Y[:split_index]
M_train       = M[:split_index]
norm_Y_train  = norm_Y[:split_index]
sigma_2_train=sigma_2[:split_index]

# validation data 
H_val       = H [split_index:]
H_noisy_val = H_noisy[split_index:] # noisy channels normalizd
Y_val       = Y[split_index:]
M_val       = M[split_index:]
norm_Y_val  = norm_Y[split_index:]
sigma_2_val = sigma_2[split_index:]

# test data 
# H_test = h_test.view(-1,U,A) #NORMALIZED # reshape channels for pga 
H_test = torch.tensor(test_data['h'],dtype=torch.complex128).view(-1,U,A) # NOT NORMALIZED
# %%####################################################################################################
###########-------------------------- E2E model WITHOUT pretraining --------------------------##########
########################################################################################################
#MPNET
import models.uPGA_model as uPGA_model
import models.Mpnet_training as Mpnet_training
optimizer='Adam'
lr = 1e-3 #not used if we dont call train_online_test_inference function
lr_constrained=1e-5 #not used if we dont call train_online_test_inference function
momentum = 0.9
epochs = 8
batch_size = 300
k = 8
batch_subsampling = 10
#Channels generation parameters
nb_channels_test = 1000
nb_channels_train=100
nb_channels_val = 500
nb_channels_per_batch=100
A=64
#imperfect config parameters
snr_in_dB = 10 
train_type = 'Online'
#scene generation params
BS_position=[-302, 42, 23.0]
f0=28e9 #HZ
sigma_ant= 0.1
#get real and nominal antenna pos
path_init=Path.cwd()/'.saved_data'
file_name = r'Data/antenna_position.npz'  
antenna_pos = np.load(path_init/file_name)
nominal_ant_positions=antenna_pos['nominal_position']
real_ant_positions=antenna_pos['real_position']
######Dictionnary parameters######
#random Azimuth angles uniformly distributed between 0 and 2*pi
DoA= np.load(path_init/f'Data/DoA.npz')['DoA']
g_vec= np.ones(A)
lambda_ =  0.010706874
mpnet = Mpnet_training.UnfoldingModel_Sim(
                BS_position,nominal_ant_positions,real_ant_positions,sigma_ant,DoA,g_vec,lambda_,snr_in_dB,lr,lr_constrained,momentum,optimizer,epochs, batch_size,k,
                batch_subsampling,train_type,f0,nb_channels_test,nb_channels_train,nb_channels_val,nb_channels_per_batch,A,L,T,noise_var
)

#PGA
hyp_mu=1e-4

num_of_iter_pga_unf = 10 
mu_unf = torch.tensor([[hyp_mu] * (2)] * num_of_iter_pga_unf, requires_grad=True)
# uPGA model defining
pga = uPGA_model.uPGA(mu_unf)


E2E_model_naive=End_to_End_model.CompositeModel(pga , mpnet.mpNet_Constrained, nominal_ant_positions,DoA,g_vec,lambda_)

# custom optimizer
optimizer=torch.optim.Adam([
     {'params': E2E_model_naive.mpnet.parameters(),'lr':1e-5},
     {'params': E2E_model_naive.pga.parameters()  ,'lr':1e-5}

])

#%% ------------------------evaluate BEFORE training -------------------------------------
sum_rate_unf, wa, wd,WA,WD,H_hat_0 = E2E_model_naive.forward(y_test,h_noisy_test,h_test,norm_y_test,k,sigma_2_test,M_test,U,A,L,T,num_of_iter_pga_unf,noise_var_DL)
sum_rate_0=evaluate_sum_rate(H_test,WA,WD,U,noise_var_DL,H_test.shape[0],10)

#%% --------------------model training----------------------------------------------------
start_time = time.time()

train_losses, valid_losses = [], []
train_NMSE=[]
best_loss=torch.inf

batch_size = 100
for i in tqdm(range(epochs)):
    for count,b in enumerate(range(0, len(H_train), batch_size)):
 

        #------Get Data-------
        h_batch             = H_train[b:b+batch_size]  # true channels not normalized
        h_noisy_batch       = H_noisy_train[b:b+batch_size] # noisy channels normalizd
        y_train_batch       = Y_train[b:b+batch_size]
        M_train_batch       = M_train[b:b+batch_size]
        norm_y_train_batch  = norm_Y_train[b:b+batch_size]
        sigma_2_train_batch        = sigma_2_train[b:b+batch_size]

        if i==0 and b==0:
            with torch.no_grad():
                # train loss
                __, wa, wd,_,_,h_hat = E2E_model_naive.forward(Y_train,H_noisy_train,H_train,norm_Y_train,k,sigma_2_train,M_train,U,A,L,T,num_of_iter_pga_unf,noise_var_DL)
                train_losses.append(sum_loss(wa, wd, h_hat.view(-1,U,A), U, len(h_hat.view(-1,U,A)),noise_var_DL))
                # validation loss
                __, wa, wd,_,_,h_hat_val  = E2E_model_naive.forward(Y_val,H_noisy_val,H_val,norm_Y_val,k,sigma_2_val,M_val,U,A,L,T,num_of_iter_pga_unf,noise_var_DL)
                valid_losses.append(sum_loss(wa, wd, h_hat_val.view(-1,U,A), U, len(h_hat_val.view(-1,U,A)),noise_var_DL))
 
        #------ Forward Pass--------------  
        sum_rate , wa, wd,_,_,_ = E2E_model_naive.forward(y_train_batch,h_noisy_batch,h_batch,norm_y_train_batch,k,sigma_2_train_batch,M_train_batch,U,A,L,T,num_of_iter_pga_unf,noise_var_DL)
        #------ Backward Pass-------------
        H_train_init=h_batch.view(-1,U,A)
        loss = sum_loss(wa, wd, H_train_init, U,  H_train_init.shape[0],noise_var_DL) 
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


    with torch.no_grad():
        # train loss
        __, wa, wd,_,_,h_hat = E2E_model_naive.forward(Y_train,H_noisy_train,H_train,norm_Y_train,k,sigma_2_train,M_train,U,A,L,T,num_of_iter_pga_unf,noise_var_DL)
        train_losses.append(sum_loss(wa, wd, h_hat.view(-1,U,A), U, len(h_hat.view(-1,U,A)),noise_var_DL))
        nmse=torch.mean(torch.sum(torch.abs(H_train-h_hat)**2,1)/torch.sum(torch.abs(H_train)**2,1))
        train_NMSE.append(nmse)
        # validation loss
        __, wa, wd,_,_,h_hat_val  = E2E_model_naive.forward(Y_val,H_noisy_val,H_val,norm_Y_val,k,sigma_2_val,M_val,U,A,L,T,num_of_iter_pga_unf,noise_var_DL)
        valid_loss=sum_loss(wa, wd, h_hat_val.view(-1,U,A), U, len(h_hat_val.view(-1,U,A)),noise_var_DL)
        valid_losses.append(valid_loss)
        if valid_loss<best_loss:
            torch.save(E2E_model_naive,save_dir/f'E2E_naive_L_{L}_T_{T}.pth')
            best_loss=valid_loss
            best_epoch=i+1
 
#end_time=time.time()
end_time = time.time()
print(f"Cell execution time: {end_time - start_time:.4f} seconds")


utils_function.plot_learning_curve("name",train_losses,valid_losses,num_of_iter_pga_unf,epochs,batch_size)

#%% ---------------------------------- Evaluation AFTER training -------------------------------------------
best_model=torch.load(save_dir/f'E2E_naive_L_{L}_T_{T}.pth')
sum_rate_unf, wa, wd,WA,WD,h_hat_test = best_model.forward(y_test,h_noisy_test,h_test,norm_y_test,k,sigma_2_test,M_test,U,A,L,T,num_of_iter_pga_unf,noise_var_DL)
sum_rate=evaluate_sum_rate(H_test,WA,WD,U,noise_var_DL,H_test.shape[0],10)
plot_sum_rate(sum_rate,'uPGA',estimator,sum_rate_0)
nmse_test=torch.mean(torch.sum(torch.abs(h_test-h_hat_test)**2,1)/torch.sum(torch.abs(h_test)**2,1))


save_sum_rate(f'E2E_cold_start',sum_rate)


#NMSE PLOT
plt.figure()
plt.plot(train_NMSE,color='red', label='E2E model')
plt.grid()
plt.title(f'NMSE Curve, Num Epochs = {epochs}, Batch Size = {batch_size} \n')
plt.xlabel('Epoch')
plt.legend(loc='best')               
# plt.savefig(save_dir/name,dpi=500)
plt.xlim(left=0)
plt.show()
print('NMSE for test data',nmse_test)