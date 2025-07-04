#%% importing libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pathlib import Path
import models.uPGA_model as uPGA_model
import utils.utils_function as utils_function
import os
from utils.utils_function import sum_loss,evaluate_sum_rate,save_sum_rate,plot_sum_rate
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%% ###############################################################################################################
########################################## TRUE CHANNEL + uPGA ###################################################
##################################################################################################################
'''This scripts executes the unfolded PGA algorithm using the true channels as input '''
Supervised=False
Constrained=True
Constrained_notConstrained="mpnet_c" if Constrained else "mpnet"
mpnet_sup_unsup="sup" if Supervised else "unsup"
#estimator='H_true

U = 4   # Num of users
L = 16 # RF chains
T = 1  # Mesures
A = 64   # Tx antennas


#%%
noise_var_list=[2e-3]
hy_p_mu_list=[8e-4]

#%%


for noise_var in noise_var_list:
    # noise_var_DL=A*noise_var
    noise_var_DL=noise_var
    print(f'Uplink noise variance={noise_var:.0e}, Downlink noise variance={noise_var_DL:.0e}, L={L}, T={T}, with true channels')

    max_batch_number  = 300

    path_init=Path.cwd()/'.saved_data'
    save_dir=path_init/f'pretrained_upga_models/{noise_var_DL:.0e}'
    os.makedirs(save_dir,exist_ok=True)
    ## %%------------------------------------------Get data-------------------------------------------------


    dataset_dir = f'Data/{noise_var:.0e}/data_var_snr/T_{T}'

    batch_idx=0

    stop=True

    H_true= torch.empty(0,U,A,dtype=torch.complex128)

    ##%% loading training data

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

        
        #concatenate
        H_true = torch.cat((H_true, h.view(-1, U, A)), dim=0) #[*, U, A]

        batch_idx += 1

        

    ##%%
    # load test data and denormalize channels
    test_data = np.load(path_init / dataset_dir/ f'test_data.npz')
    M_test_data=np.load(path_init/'Data'/f'Measurement_matrix/L_{L}_T_{T}'/'test.npz')

    h_test=torch.tensor(test_data['h'],dtype=torch.complex128)
    h_noisy_test=torch.tensor(test_data['h_noisy'],dtype=torch.complex128)
    M_test = torch.tensor(M_test_data['M_test'],dtype=torch.complex128).to(device)
    M_tilde_test = torch.tensor(M_test_data['M_tilde_test'],dtype=torch.complex128).to(device)
    #observed signal:
    y_test  =torch.matmul((h_noisy_test).unsqueeze(1),torch.conj(M_tilde_test)).squeeze().to(device)

    ##%%-------------------------------Get train, validation and test data -------------------------------------------------
    
    # train data 
    train_ratio=0.8
    split_index = int(len(H_true) * train_ratio)
    H_true_train    = H_true [:split_index].to(device)

    # validation data 
    H_true_val      = H_true [split_index:] # int(valid_size/U)].to(device)

    # test data 
    H_true_test     = h_test.view(-1,U,A).to(device)

    #--------------------------------------------------------------------------------------------------
    # parameters defining
    hyp_mu=1e-3
    num_of_iter_upga = 10 
    mu_unf = torch.tensor([[hyp_mu] * (2)] * num_of_iter_upga, requires_grad=True)
    lr=1e-5
    # uPGA model defining
    unfolded_model = uPGA_model.uPGA(mu_unf)

    #optimizer
    optimizer = torch.optim.Adam(unfolded_model.parameters(), lr=lr)
    ##%%-----------------------evaluating model BEFORE training--------------------------------------
    s,_,_,WA,WD = unfolded_model.forward(H_true_test, U, L,  num_of_iter_upga,noise_var_DL)
    sum_rate_0= evaluate_sum_rate(H_true_test, WA,WD,U,noise_var_DL,H_true_test.shape[0],num_of_iter_upga)

    ##%%---------------------------------------training-----------------------------------------------
    epochs = 30
    batch_size_upga = 500 # batch size

    train_losses, valid_losses = [], []
    best_loss=torch.inf
    for i in tqdm(range(epochs)):
        
        for b in range(0, len(H_true_train), batch_size_upga):
            H_true_batched  =   H_true_train          [b:b+batch_size_upga].to(device)

            if i==0 and b==0:
                with torch.no_grad():
                    # train loss
                    sum_rate_est, wa, wd,_,_ = unfolded_model.forward(H_true_train, U, L,  num_of_iter_upga,noise_var_DL)
                    train_losses.append(sum_loss(wa, wd, H_true_train, U,  H_true_train.shape[0],noise_var_DL))
                    # validation loss
                    __, wa, wd,_,_ = unfolded_model.forward(H_true_val, U, L,  num_of_iter_upga,noise_var_DL)
                    valid_loss=sum_loss(wa, wd, H_true_val, U,  H_true_val.shape[0],noise_var_DL)
                    valid_losses.append(valid_loss)


        ################################## channel estimation + uPGA #####################################################
            sum_train, wa, wd , _,_ = unfolded_model.forward(H_true_batched, U, L,  num_of_iter_upga,noise_var_DL)

            loss = sum_loss(wa, wd, H_true_batched, U, batch_size_upga, noise_var_DL)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
 
        with torch.no_grad():
            # train loss
            sum_rate_est, wa, wd,_,_ = unfolded_model.forward(H_true_train, U, L,  num_of_iter_upga,noise_var_DL)
            train_losses.append(sum_loss(wa, wd, H_true_train, U,  H_true_train.shape[0],noise_var_DL))

            # validation loss
            __, wa, wd,_,_ = unfolded_model.forward(H_true_val, U, L,  num_of_iter_upga,noise_var_DL)
            valid_loss=sum_loss(wa, wd, H_true_val, U,  H_true_val.shape[0],noise_var_DL)
            valid_losses.append(valid_loss)
            if valid_loss<best_loss:
                torch.save(unfolded_model,save_dir/f'uPGA_true_channels_L_{L}_T_{T}.pth')
                best_loss=valid_loss
                best_epoch=i+1

    #plotting learning curve
    name=f'Loss_curve_uPGA_true_channels_L_{L}_T_{T}'
    utils_function.plot_learning_curve(name,train_losses,valid_losses,num_of_iter_upga,epochs,batch_size_upga)
    ##%%

    best_model=torch.load(save_dir/f'uPGA_true_channels_L_{L}_T_{T}.pth')
    s,_,_,WA,WD = best_model.forward(H_true_test, U, L,  num_of_iter_upga,noise_var_DL)
    sum_rate_est= evaluate_sum_rate(H_true_test, WA,WD,U,noise_var_DL,H_true_test.shape[0],num_of_iter_upga)
    save_sum_rate(f'uPGA_true_channels',sum_rate_est,noise_var_DL,L,T)
    plot_sum_rate(sum_rate_est,'uPGA','True channels',A,U,L,T,sum_rate_0)
