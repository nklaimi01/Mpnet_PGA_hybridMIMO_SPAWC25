#%% libraries
import numpy as np
import torch
from tqdm import trange,tqdm
from pathlib import Path
import utils.sparse_recovery as sparse_recovery
import os
from utils.lmmse_hybrid import LMMSE_estimation
import utils.generate_steering as generate_steering
import torch
from pathlib import Path
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%% functions
def save_estimation_mpnet(model, data_file, data_pred_mpnet_unsup, batch_size, k, T, L,noise_var):
    
    path_init = Path.cwd()/'.saved_data'

    # Load trained model
    mpnet=torch.load(model)
    

    ####################################           test                   ############################################
    ####################################           Data                   ############################################
    
    ## estimate test channels 

    # load test data 
    
    test_data = np.load(path_init/data_file/'test_data.npz')
                    
    h_test        =    torch.tensor(test_data['h'],dtype=torch.complex128).to(device)         
    h_noisy_test  =    torch.tensor(test_data['h_noisy'],dtype=torch.complex128).to(device)       
    sigma_2_test  =    torch.tensor(test_data['sigma_2']).to(device)      
    nb_BS_antenna=h_test.shape[1]
    # Get Measurement matrix
    M_data=np.load(path_init/'Data'/f'Measurement_matrix/L_{L}_T_{T}'/'test.npz')
    M_test = torch.tensor(M_data['M_test'],dtype=torch.complex128).to(device) #shape [*,A,TL]
    M_tilde_test = torch.tensor(M_data['M_tilde_test'],dtype=torch.complex128).to(device) #shape [*,A,TL]

    # the observed signal
    h_noisy_M_test = torch.matmul(h_noisy_test.unsqueeze(1), torch.conj(M_tilde_test)).squeeze()
    ## preprocessing 
    norm_noisy_M_test = torch.norm(h_noisy_M_test, p=2, dim=1)
    h_noisy_M_test = h_noisy_M_test / norm_noisy_M_test[:, None]
    # normalize channels 
    h_noisy_test   = h_noisy_test / (norm_noisy_M_test[:,None]/np.sqrt(L))
    h_test         = h_test / (norm_noisy_M_test[:,None]/np.sqrt(T*L))



    ######################################  MPNET estimation #############################################################

    # Mpnet Estimation
    sigma_norm_test = torch.sqrt(sigma_2_test) / norm_noisy_M_test

    _, _, est_chan = mpnet(h_noisy_M_test, h_noisy_test, M_test, L, T, k, sigma_norm_test, 2)
    est_chan = est_chan.detach().cpu().numpy()[:,:nb_BS_antenna]

    # Save estimation
    est_c_mpnet = {'channels': est_chan}

    # Cost function
    nmse_test =torch.mean(torch.sum(torch.abs(h_test.cpu()-est_chan)**2,1)/torch.sum(torch.abs(h_test.cpu())**2,1))
    tqdm.write(f'NMSE MpNet Test data : {nmse_test}')

    # Save data
    save_dir=path_init / data_pred_mpnet_unsup
    os.makedirs(save_dir, exist_ok=True)  # Creates folder if it doesn't exist
    np.savez(save_dir / f'test.npz', **est_c_mpnet)


    ####################################           train                  ############################################
    ####################################           Data                   ############################################



    for batch_idx in trange(batch_size):

        data = np.load(path_init / data_file / f'batch_{batch_idx}.npz')
        h = torch.tensor(data['h'], dtype=torch.complex128).to(device)
        h_noisy = torch.tensor(data['h_noisy'], dtype=torch.complex128).to(device)
        sigma_2 = torch.tensor(data['sigma_2']).to(device)
        # Get Measurement matrix
        M_data=np.load(path_init/'Data'/f'Measurement_matrix/L_{L}_T_{T}'/f'batch_{batch_idx}.npz')
        M = torch.tensor(M_data['M_train'],dtype=torch.complex128).to(device)
        M_tilde = torch.tensor(M_data['M_tilde_train'],dtype=torch.complex128).to(device)
        # the observed signal:
        h_noisy_M = torch.matmul(h_noisy.unsqueeze(1), torch.conj(M_tilde)).squeeze().to(device)

        norm_noisy_M = torch.norm(h_noisy_M, p=2, dim=1).to(device)
        h_noisy_M = h_noisy_M / norm_noisy_M[:, None]
        h = h / (norm_noisy_M[:, None]/np.sqrt(T*L))
        h_noisy=h_noisy/(norm_noisy_M[:, None]/np.sqrt(L))


        ###################################### MPNET estimation #############################################################

        # Mpnet Estimation
        sigma_norm = torch.sqrt(sigma_2) / norm_noisy_M

        _, _, est_chan = mpnet(h_noisy_M, h_noisy, M, L, T, k, sigma_norm, 2)
        est_chan = est_chan.detach().cpu().numpy()[:,:nb_BS_antenna]

        # Save estimation
        est_c = {'channels': est_chan}

        # Cost function
        nmse_train =torch.mean(torch.sum(torch.abs(h.cpu()-est_chan)**2,1)/torch.sum(torch.abs(h.cpu())**2,1))
        # tqdm.write(f'NMSE MpNet : {nmse_train}')

        # Save data
        save_dir=path_init / data_pred_mpnet_unsup
        os.makedirs(save_dir,exist_ok=True)
        np.savez( save_dir/ f'batch_{batch_idx}.npz', **est_c)


def save_estimation_MP_LMMSE(data_file, data_pred_MP_nominal, data_pred_MP_real, data_pred_LMMSE, batch_size, k, T, L, dict_nominal, dict_real,nb_BS_antenna,noise_var):

    
    path_init = Path.cwd()/'.saved_data'

    # Prepare dictionaries on GPU
    dict_nominal = dict_nominal.to(device)
    dict_real = dict_real.to(device)

    ####################################           test                   ############################################
    ####################################           Data                   ############################################
    
    ## estimate test channels 

        # load test data 
    
    test_data = np.load(path_init/data_file/'test_data.npz')
    M_data=np.load(path_init/'Data'/f'Measurement_matrix/L_{L}_T_{T}'/'test.npz')
                    
    h_test        =    torch.tensor(test_data['h'],dtype=torch.complex128).to(device)         
    h_noisy_test  =    torch.tensor(test_data['h_noisy'],dtype=torch.complex128).to(device)       
    sigma_2_test  =    torch.tensor(test_data['sigma_2']).to(device)      
    # Get Measurement matrix
    M_test = torch.tensor(M_data['M_test'],dtype=torch.complex128).to(device) #shape [*,A,TL]
    M_tilde_test = torch.tensor(M_data['M_tilde_test'],dtype=torch.complex128).to(device) #shape [*,A,TL]
    #the observed signal:
    y_test = torch.matmul(h_noisy_test.unsqueeze(1), torch.conj(M_tilde_test)).squeeze()

    ## preprocessing 
    norm_y_test     =torch.norm(y_test,p=2,dim=1).to(device)
    # normalize channels 
    y_test  =y_test/ norm_y_test[:,None]
    h_noisy_test   = h_noisy_test / (norm_y_test[:,None]/np.sqrt(L))
    h_test         = h_test / (norm_y_test[:,None]/np.sqrt(T*L))
    sigma_2_test = torch.sqrt(sigma_2_test) / norm_y_test
    SC2 = pow(sigma_2_test,2) * nb_BS_antenna * L  
    ######################################### Mp real Estimation ##################################################
    out_chans = torch.zeros_like(h_test, dtype=torch.complex128).to(device)

    for i in range(h_test.shape[0]):
            dict_real_M = torch.matmul(torch.conj(M_test.mT), dict_real.to(device)).to(device)
            dict_real_M = dict_real_M.squeeze(0)
            out_chans[i, :] = sparse_recovery.mp(h_noisy_test[i, :], y_test[i, :], dict_real.to(device), dict_real_M, k, False, SC2[i])

    # Save estimation
    est_c_mp_r = {'channels': out_chans.cpu().numpy()}

    # Cost function
    NMSE_mp_r = torch.mean(torch.linalg.norm(h_test - out_chans.cpu(), axis=1) ** 2 / torch.linalg.norm(h_test, axis=1) ** 2)
    tqdm.write(f'NMSE MP REAL test data : {NMSE_mp_r}')

    save_dir=path_init / data_pred_MP_real
    os.makedirs(save_dir, exist_ok=True)  # Creates folder if it doesn't exist
    np.savez( save_dir/'test.npz', **est_c_mp_r)
    
    ######################################### Mp nominal Estimation ##################################################

    out_chans = torch.zeros_like(h_test, dtype=torch.complex128).to(device)

    for i in range(h_test.shape[0]):
            dict_nominal_M = torch.matmul(torch.conj(M_test.mT), dict_nominal.to(device)).to(device)
            dict_nominal_M = dict_nominal_M.squeeze(0)
            out_chans[i, :] = sparse_recovery.mp(h_noisy_test[i, :], y_test[i, :], dict_nominal, dict_nominal_M, k, False, SC2[i])

    # Save estimation
    est_c_mp_n = {'channels': out_chans.cpu().numpy()}

    # Cost function
    NMSE_mp_n = torch.mean(torch.linalg.norm(h_test - out_chans.cpu(), axis=1) ** 2 / torch.linalg.norm(h_test, axis=1) ** 2)
    tqdm.write(f'NMSE MP NOMINAL test data : {NMSE_mp_n}')

    save_dir=path_init / data_pred_MP_nominal
    os.makedirs(save_dir, exist_ok=True)  # Creates folder if it doesn't exist    
    np.savez(save_dir / 'test.npz', **est_c_mp_n)

    ######################################### LMMSE estimation####################################################
    h_hat_lmmse=LMMSE_estimation(h_test,h_noisy_test,sigma_2_test,M_test,M_tilde_test)
    est_c_lmmse = {'channels': h_hat_lmmse.cpu().numpy()}
    # Cost function
    NMSE_mp_n = torch.mean(torch.linalg.norm(h_test - h_hat_lmmse, axis=1) ** 2 / torch.linalg.norm(h_test, axis=1) ** 2)
    tqdm.write(f'NMSE LMMSE test data : {NMSE_mp_n}')

    save_dir=path_init / data_pred_LMMSE
    os.makedirs(save_dir, exist_ok=True) 
    np.savez(save_dir / 'test.npz', **est_c_lmmse)
        
   

    ####################################           train                  ############################################
    ####################################           Data                   ############################################



    for batch_idx in trange(batch_size):

        data = np.load(path_init / data_file / f'batch_{batch_idx}.npz')
        M_data=np.load(path_init/'Data'/f'Measurement_matrix/L_{L}_T_{T}'/f'batch_{batch_idx}.npz')
        h = torch.tensor(data['h'], dtype=torch.complex128).to(device)
        h_noisy = torch.tensor(data['h_noisy'], dtype=torch.complex128).to(device)
        sigma_2 = torch.tensor(data['sigma_2']).to(device)
        # Get Measurement matrix
        M = torch.tensor(M_data['M_train'],dtype=torch.complex128).to(device)
        M_tilde = torch.tensor(M_data['M_tilde_train'],dtype=torch.complex128).to(device)
        # the observed signal:
        y_train = torch.matmul(h_noisy.unsqueeze(1), torch.conj(M_tilde)).squeeze().to(device)

        #NORMALIZATION
        norm_y_train = torch.norm(y_train, p=2, dim=1).to(device)
        y_train = y_train / norm_y_train[:, None]
        h = h / (norm_y_train[:, None]/np.sqrt(T*L))
        h_noisy=h_noisy/(norm_y_train[:, None]/np.sqrt(L))
        sigma_2 = torch.sqrt(sigma_2) / norm_y_train
        SC2 = pow(sigma_2,2) * nb_BS_antenna * L 
        ########################################### Mp nominal Estimation ##############################################
        out_chans = torch.zeros_like(h, dtype=torch.complex128).to(device)



        for i in range(h.shape[0]):
            dict_nominal_M = torch.matmul(torch.conj(M.mT), dict_nominal)
            dict_nominal_M = dict_nominal_M.squeeze(0)
            out_chans[i, :] = sparse_recovery.mp(h_noisy[i, :], y_train[i, :], dict_nominal, dict_nominal_M, k, False, SC2[i])

        # Save estimation
        est_c_mp = {'channels': out_chans.cpu().numpy()}
        # Cost function
        NMSE_mp_n = torch.mean(torch.linalg.norm(h - out_chans.cpu(), axis=1) ** 2 / torch.linalg.norm(h, axis=1) ** 2)
        # tqdm.write(f'NMSE MP NOMINAL : {NMSE_mp_n}')

        save_dir=path_init / data_pred_MP_nominal
        os.makedirs(save_dir,exist_ok=True)
        np.savez(save_dir / f'batch_{batch_idx}.npz', **est_c_mp)
        
        
        ######################################### Mp real Estimation ##################################################
        out_chans = torch.zeros_like(h, dtype=torch.complex128).to(device)

        for i in range(h.shape[0]):
            dict_real_M = torch.matmul(torch.conj(M.mT), dict_real).type(torch.complex128).to(device)
            dict_real_M=dict_real_M.squeeze(0)
            out_chans[i, :] = sparse_recovery.mp(h_noisy[i, :], y_train[i, :], dict_real, dict_real_M, k, False, SC2[i])

        # Save estimation
        est_c_mp = {'channels': out_chans.cpu().numpy()}

        # Cost function
        NMSE_mp_r = torch.mean(torch.linalg.norm(h - out_chans.cpu(), axis=1) ** 2 / torch.linalg.norm(h, axis=1) ** 2)
        # tqdm.write(f'NMSE MP REAL : {NMSE_mp_r}')

        save_dir=path_init / data_pred_MP_real
        os.makedirs(save_dir,exist_ok=True)
        np.savez( save_dir/ f'batch_{batch_idx}.npz', **est_c_mp)


        ######################################### LMMSE estimation####################################################
        h_hat_lmmse=LMMSE_estimation(h,h_noisy,sigma_2,M,M_tilde)
        est_c_lmmse = {'channels': h_hat_lmmse.cpu().numpy()}
        # Cost function
        NMSE_mp_n = torch.mean(torch.linalg.norm(h - h_hat_lmmse, axis=1) ** 2 / torch.linalg.norm(h, axis=1) ** 2)
        # tqdm.write(f'NMSE LMMSE : {NMSE_mp_n}')

        save_dir=path_init / data_pred_LMMSE
        os.makedirs(save_dir, exist_ok=True) 
        np.savez(save_dir / f'batch_{batch_idx}.npz', **est_c_lmmse)


# #%% ------------------------------------- test code ---------------------------------------------------------------------
# # model characteristics 
# noise_var= 6e-3
# # RF chains 
# L = 16
# T = 1
# print(f'saving channel estamtions with noise variance={noise_var:.0e}, L={L}, T={T}')
# nb_BS_antenna=64
# batch_size = 300
# k = 8
# data_file=f'Data/{noise_var:.0e}/data_var_snr/T_{T}'

# #%% ----------------------------------mpNet--------------------------------------------------------------------

# data_pred_mpnet=f'Channel_estimation/{noise_var:.0e}/L_{L}_T_{T}/mpnet_c_unsup'
# model_=f'pretrained_mpnet_models/{noise_var:.0e}/mpnet_c_unsup_L_{L}_T_{T}.pth'
# save_estimation_mpnet(model_, data_file, data_pred_mpnet, batch_size, k, T, L,noise_var)
# data_pred_mpnet=f'Channel_estimation/{noise_var:.0e}/L_{L}_T_{T}/mpnet_sup'
# model_=f'pretrained_mpnet_models/{noise_var:.0e}/mpnet_sup_L_{L}_T_{T}.pth'
# save_estimation_mpnet(model_, data_file, data_pred_mpnet, batch_size, k, T, L,noise_var)

# #%% ---------------------------------- LMMSE / MP --------------------------------------------------------------------
# #get real and nominal antenna pos
# path_init=Path.cwd()/'.saved_data'
# antenna_pos = np.load(path_init/'Data/datasionna/antenna_position.npz')
# nominal_ant_positions=antenna_pos['nominal_position']
# real_ant_positions=antenna_pos['real_position']

# ######Dictionnary parameters######
# #random Azimuth angles uniformly distributed between 0 and 2*pi
# DoA= np.load(path_init/f'Data/DoA.npz')['DoA']
# g_vec= np.ones(nb_BS_antenna)
# lambda_ =  0.010706874
# #--------------channel_estimations--------------

# data_pred_MP_nominal=f'Channel_estimation/{noise_var:.0e}/L_{L}_T_{T}/nominal'
# data_pred_MP_real=f'Channel_estimation/{noise_var:.0e}/L_{L}_T_{T}/real'
# data_pred_LMMSE=f'Channel_estimation/{noise_var:.0e}/L_{L}_T_{T}/lmmse'
# batch_size_est=300

# # construct nominal dictionary 
# dict_nominal=generate_steering.steering_vect_c(torch.tensor(nominal_ant_positions).type(torch.FloatTensor),
#                                                        torch.tensor(DoA).type(torch.FloatTensor),
#                                                        torch.tensor(g_vec),
#                                                        lambda_)
# # construct real dictionary 
# dict_real   =generate_steering.steering_vect_c(torch.tensor(real_ant_positions).type(torch.FloatTensor),
#                                                        torch.tensor(DoA).type(torch.FloatTensor),
#                                                        torch.tensor(g_vec),
#                                                        lambda_)


# save_estimation_MP_LMMSE(data_file, data_pred_MP_nominal, data_pred_MP_real, data_pred_LMMSE, batch_size, k, T, L, dict_nominal, dict_real,nb_BS_antenna,noise_var)

