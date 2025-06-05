#%% importing libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pathlib import Path
import models.ProjGradAscent as ProjGradAscent
import utils.utils_function as utils_function
import os
from utils.utils_function import sum_loss,evaluate_sum_rate
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'
#%% functions: 
# ploting the results
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


def save_sum_rate(file_name,sum_rate_,noise_var_DL):
    
    # save sum rate
    sum_rate={}
    sum_rate[f'{file_name}']=sum_rate_.detach().numpy()
    save_dir=path_init / 'sumRate'/f'{noise_var_DL:.0e}/L_{L}_T_{T}'
    os.makedirs(save_dir,exist_ok=True)
    np.savez(save_dir/f'{file_name}', **sum_rate)

# %%
Supervised=False
Constrained=True
Constrained_notConstrained="mpnet_c" if Constrained else "mpnet"
mpnet_sup_unsup="sup" if Supervised else "unsup"
estimator=f'{Constrained_notConstrained}_{mpnet_sup_unsup}' 
# estimator='lmmse'

U = 4   # Num of users
L = 16 # RF chains
T = 1  # Mesures
A = 64   # Tx antennas
#%%
noise_var_list=[2e-3]
hy_p_mu_list=[8e-4]

# noise_var_list=[6e-04,2e-3, 6e-03, 2e-02, 6e-02, 2e-01]
# hy_p_mu_list=[1e-3,1e-3,1e-3,5e-3,5e-3,5e-2]
#%% ########################################################################################################
#######------------------------------ estimator + uPGA ---------------------------------------------########
############################################################################################################

for j,noise_var in enumerate(noise_var_list):
    # noise_var_DL=A*noise_var # Downlink noise variance
    noise_var_DL=noise_var # Downlink noise variance
    print(f'Uplink noise variance={noise_var:.0e}, Downlink noise variance={noise_var_DL:.0e}, L={L}, T={T}, estimator={estimator}')

    max_batch_number  = 300

    path_init=Path.cwd()/'.saved_data'
    save_dir=path_init/f'pretrained_upga_models/{noise_var_DL:.0e}'
    os.makedirs(save_dir,exist_ok=True)
    ## %%------------------------------------------Get data-------------------------------------------------


    dataset_dir = f'Data/{noise_var:.0e}/data_var_snr/T_{T}'
    estimation_dir = f'Channel_estimation/{noise_var:.0e}/L_{L}_T_{T}/{estimator}'

    batch_idx=0

    stop=True

    H_est= torch.empty(0,U,A,dtype=torch.complex128)
    H_true= torch.empty(0,U,A,dtype=torch.complex128)
    NMSE_list=[]

    ##%% loading training data

    '''
    h: true channels without noise
    h_noisy: noisy channels 
    '''
    while batch_idx<max_batch_number:
        # Load true channels
        train_data = np.load(path_init / dataset_dir/ f'batch_{batch_idx}.npz')
        M_data=np.load(path_init/'Data'/f'Measurement_matrix/L_{L}_T_{T}'/f'batch_{batch_idx}.npz')

        h = torch.tensor(train_data['h'], dtype=torch.complex128)
        h_noisy = torch.tensor(train_data['h_noisy'],dtype=torch.complex128).to(device)       
        
        M = torch.tensor(M_data['M_train'],dtype=torch.complex128).to(device)
        M_tilde = torch.tensor(M_data['M_tilde_train'],dtype=torch.complex128).to(device)
        #observed signal:
        y_train  =torch.matmul((h_noisy).unsqueeze(1),torch.conj(M_tilde)).squeeze().to(device)


        # denorm_factor  = torch.norm(h_noisy,p=2,dim=1)[:, None]/np.sqrt(T)
        denorm_factor  = torch.norm(y_train,p=2,dim=1)[:, None]/np.sqrt(T*L)

        # Load Mpnet Estimated channels
        est_data = np.load(path_init / estimation_dir / f'batch_{batch_idx}.npz')   
        est_channels = torch.tensor(est_data['channels'], dtype=torch.complex128)[:,:A]
        est_channels = est_channels * denorm_factor
        
        #concatenate
        H_true = torch.cat((H_true, h.view(-1, U, A)), dim=0) #[*, U, A]
        H_est = torch.cat((H_est, est_channels.view(-1, U, A)), dim=0)


        NMSE= torch.mean(torch.linalg.norm(h - est_channels[:,:A].cpu(), axis=1) ** 2 / torch.linalg.norm(h, axis=1) ** 2)
        NMSE_list.append(NMSE)

        batch_idx += 1

        

    ##%%
    # load test data and denormalize channels
    test_data = np.load(path_init / dataset_dir/ f'test_data.npz')
    est_test_data = np.load(path_init / estimation_dir / f'test.npz') 
    M_test_data=np.load(path_init/'Data'/f'Measurement_matrix/L_{L}_T_{T}'/'test.npz')

    h_test=torch.tensor(test_data['h'],dtype=torch.complex128)
    h_noisy_test=torch.tensor(test_data['h_noisy'],dtype=torch.complex128)
    M_test = torch.tensor(M_test_data['M_test'],dtype=torch.complex128).to(device)
    M_tilde_test = torch.tensor(M_test_data['M_tilde_test'],dtype=torch.complex128).to(device)
    #observed signal:
    y_test  =torch.matmul((h_noisy_test).unsqueeze(1),torch.conj(M_tilde_test)).squeeze().to(device)
    # denorm_factor  = torch.norm(h_noisy,p=2,dim=1)[:, None]/np.sqrt(T)
    denorm_factor_test  = torch.norm(y_test,p=2,dim=1)[:, None]/np.sqrt(T*L)

    # denormalize estimations
    h_est = torch.tensor(est_test_data['channels'], dtype=torch.complex128)[:,:A]
    h_est     = h_est   * denorm_factor_test

    # visualize channel estimation quality
    NMSE_test= torch.mean(torch.linalg.norm(h_test - h_est.cpu(), axis=1) ** 2 / torch.linalg.norm(h_test, axis=1) ** 2)
    print(NMSE_test)
    plt.plot(NMSE_list)

    ##%%-------------------------------Get train, validation and test data -------------------------------------------------
    
    # train data 
    train_ratio=0.8
    split_index = int(len(H_true) * train_ratio)
    H_true_train    = H_true [:split_index].to(device)
    H_est_train   = H_est [:split_index].to(device)

    # validation data 
    H_true_val      = H_true [split_index:] # int(valid_size/U)].to(device)
    H_est_val     = H_est[split_index:] # int(valid_size/U)].to(device)

    # test data 
    H_true_test     = h_test.view(-1,U,A).to(device)
    H_est_test    = h_est.view(-1,U,A).to(device)

    ##%% -----------------------------------Unfolded PGA ------------------------------------------
    # parameters defining
    hyp_mu=hy_p_mu_list[j]
    num_of_iter_pga_unf = 10 
    mu_unf = torch.tensor([[hyp_mu] * (2)] * num_of_iter_pga_unf, requires_grad=True)
    lr=1e-4
    # uPGA model defining
    unfolded_model = ProjGradAscent.ProjGA(mu_unf)

    #optimizer
    optimizer = torch.optim.Adam(unfolded_model.parameters(), lr=lr)
    scheduler= torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)
    ##%%-----------------------evaluating model BEFORE training--------------------------------------
    s,_,_,WA,WD = unfolded_model.forward(H_est_test, U, L,  num_of_iter_pga_unf,noise_var_DL)
    sum_rate_0= evaluate_sum_rate(H_true_test, WA,WD,U,noise_var_DL,H_true_test.shape[0],num_of_iter_pga_unf)

    ##%%---------------------------------------training-----------------------------------------------
    epochs = 20
    batch_size_upga = 500 # batch size

    train_losses, valid_losses = [], []
    best_loss=torch.inf
    for i in tqdm(range(epochs)):
        
        for b in range(0, len(H_true_train), batch_size_upga):
            H_est_batched =   H_est_train        [b:b+batch_size_upga].to(device)
            H_true_batched  =   H_true_train          [b:b+batch_size_upga].to(device)

            if i==0 and b==0:
                with torch.no_grad():
                    # train loss
                    sum_rate_est, wa, wd,_,_ = unfolded_model.forward(H_est_train, U, L,  num_of_iter_pga_unf,noise_var_DL)
                    train_losses.append(sum_loss(wa, wd, H_est_train, U,  H_est_train.shape[0],noise_var_DL))
                    # validation loss
                    __, wa, wd,_,_ = unfolded_model.forward(H_est_val, U, L,  num_of_iter_pga_unf,noise_var_DL)
                    valid_loss=sum_loss(wa, wd, H_est_val, U,  H_est_val.shape[0],noise_var_DL)
                    valid_losses.append(valid_loss)


        ################################## channel estimation + uPGA #####################################################
            sum_train, wa, wd , _,_ = unfolded_model.forward(H_est_batched, U, L,  num_of_iter_pga_unf,noise_var_DL)
            if Supervised:
                loss = sum_loss(wa, wd, H_true_batched, U, batch_size_upga, noise_var_DL)
            else:
                loss = sum_loss(wa, wd, H_est_batched, U, batch_size_upga, noise_var_DL)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step() # Update the learning rate using the scheduler

        

        with torch.no_grad():
            # train loss
            sum_rate_est, wa, wd,_,_ = unfolded_model.forward(H_est_train, U, L,  num_of_iter_pga_unf,noise_var_DL)
            train_losses.append(sum_loss(wa, wd, H_est_train, U,  H_est_train.shape[0],noise_var_DL))

            # validation loss
            __, wa, wd,_,_ = unfolded_model.forward(H_est_val, U, L,  num_of_iter_pga_unf,noise_var_DL)
            valid_loss=sum_loss(wa, wd, H_est_val, U,  H_est_val.shape[0],noise_var_DL)
            valid_losses.append(valid_loss)
            if valid_loss<best_loss:
                torch.save(unfolded_model,save_dir/f'pga_{estimator}_L_{L}_T_{T}.pth')
                best_loss=valid_loss
                best_epoch=i+1

    #plotting learning curve
    name=f'Loss_curve_upga_{estimator}_L_{L}_T_{T}'
    utils_function.plot_learning_curve(name,train_losses,valid_losses,num_of_iter_pga_unf,epochs,batch_size_upga)


    ##%% --------------- evaluate best model ----------------------------
    # num_of_iter_pga_unf=10
    best_model=torch.load(save_dir/f'pga_{estimator}_L_{L}_T_{T}.pth')
    s,_,_,WA,WD = best_model.forward(H_est_test, U, L,  num_of_iter_pga_unf,noise_var_DL)
    sum_rate_est= evaluate_sum_rate(H_true_test, WA,WD,U,noise_var_DL,H_true_test.shape[0],num_of_iter_pga_unf)
    save_sum_rate(f'unf_est_{estimator}',sum_rate_est,noise_var_DL)
    plot_sum_rate(sum_rate_est,'uPGA',estimator,sum_rate_0)

