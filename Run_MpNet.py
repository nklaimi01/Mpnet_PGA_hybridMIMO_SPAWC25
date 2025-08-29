#%%
import numpy as np
import time
import models.Mpnet_training as Mpnet_training
import torch
from pathlib import Path
import os
import matplotlib.pyplot as plt

path_init=Path.cwd()/'.saved_data'
save_dir=path_init/'paper_mpnet_models'
#%%
noise_var= 2e-2
# RF chains 
L = 16
T = 1
print(f"MpNet training with noise_var={noise_var:.0e}, L={L}, T={T}")
#%% ######model hyperparams######
optimizer='Adam'
lr = 1e-3
lr_constrained=1e-5
momentum = 0.9
epochs = 1
batch_size = 300
k = 8
batch_subsampling = 10

#Channels generation parameters
nb_channels_test = 1000
nb_channels_train=100
nb_channels_val = 500
nb_channels_per_batch=100
nb_BS_antenna=64


#imperfect config parameters
snr_in_dB = 10 
train_type = 'Online'

#scene generation params
BS_position=[-302, 42, 23.0]
f0=28e9 #HZ

sigma_ant= 0.1

#get real and nominal antenna pos
file_name = r'Data/antenna_position.npz'  
antenna_pos = np.load(path_init/file_name)
nominal_ant_positions=antenna_pos['nominal_position']
real_ant_positions=antenna_pos['real_position']
######Dictionnary parameters######
#random Azimuth angles uniformly distributed between 0 and 2*pi
DoA= np.load(path_init/f'Data/DoA.npz')['DoA']
g_vec= np.ones(nb_BS_antenna)
lambda_ =  0.010706874

#%% --------------------------------- unsupervised ---------------------------------
model = Mpnet_training.UnfoldingModel_Sim(
                 BS_position,
                 nominal_ant_positions,
                 real_ant_positions,
                 sigma_ant,
                 DoA,
                 g_vec,
                 lambda_,
                 snr_in_dB,                  
                 lr,
                 lr_constrained,
                 momentum,                 
                 optimizer,
                 epochs, batch_size,
                 k, 
                 batch_subsampling,
                 train_type,
                 f0,
                 nb_channels_test,
                 nb_channels_train,
                 nb_channels_val,
                 nb_channels_per_batch,
                 nb_BS_antenna,
                 L,
                 T,
                 noise_var
)

start_time = time.time()

model.train_online_test_inference(noise_var,supervised=False)  

end_time = time.time()
print(f"Cell execution time: {end_time - start_time:.4f} seconds")


#%% ------------------- Supervised ---------------------------------
model_sup = Mpnet_training.UnfoldingModel_Sim(
                 BS_position,
                 nominal_ant_positions,
                 real_ant_positions,
                 sigma_ant,
                 DoA,
                 g_vec,
                 lambda_,
                 snr_in_dB,                  
                 lr,
                 lr_constrained,
                 momentum,                 
                 optimizer,
                 epochs, batch_size,
                 k, 
                 batch_subsampling,
                 train_type,
                 f0,
                 nb_channels_test,
                 nb_channels_train,
                 nb_channels_val,
                 nb_channels_per_batch,
                 nb_BS_antenna,
                 L,
                 T,
                 noise_var
)

start_time = time.time()

model_sup.train_online_test_inference(noise_var,supervised=True)  

end_time = time.time()
print(f"Cell execution time: {end_time - start_time:.4f} seconds")
#%%
# save model for results in paper
os.makedirs(save_dir,exist_ok=True)
torch.save(model,save_dir/f'model_{noise_var:.0e}_L_{L}_T_{T}.pth')
torch.save(model_sup,save_dir/f'model_sup_{noise_var:.0e}_L_{L}_T_{T}.pth')

#%% ############################################################################################
###########---------------------NMSE vs nb of seen channels ------------------##################
################################################################################################

model=torch.load(save_dir/f'model_{noise_var:.0e}_L_{L}_T_{T}.pth')
model_sup=torch.load(save_dir/f'model_sup_{noise_var:.0e}_L_{L}_T_{T}.pth')
# plot figure
plt.rcParams['text.usetex'] = True
plt.figure()

# Ensure nominal MP values are set correctly
NMSE_mp_nominal = np.ones_like(model.NMSE_mp_nominal) * model.NMSE_mpnet_test[0]  # Same for both models

# Define colors
mpnet_sup_color = 'tab:blue'        # mpNet Unsup & Sup
mpnet_unsup_color = 'tab:orange'  # mpNet Constrained Unsup & Sup
nominal_color = 'tab:purple'    # MP Nominal
lmmse_color = 'tab:red'         # LMMSE
mp_real_color = 'tab:green'     # MP Real

# lower bounds
x_reduced = np.unique(np.concatenate([np.arange(0, len(model.NMSE_mpnet_test), step=5), [len(model.NMSE_mpnet_test) - 1]])) 
plt.plot(x_reduced,model.NMSE_lmmse[x_reduced], 'x-', color=lmmse_color, linewidth=1.5, label='LMMSE')
plt.plot(NMSE_mp_nominal, ':', color=nominal_color, linewidth=1.5, label='MP (Nominal dictionary)')  

# Plot results
plt.plot(model.NMSE_mpnet_test, '.-', color=mpnet_unsup_color, linewidth=1, label='mpNet Unsupervised')  
plt.plot(model_sup.NMSE_mpnet_test, '--', color=mpnet_sup_color, linewidth=1, label='mpNet Supervised')  

plt.plot(model.NMSE_mpnet_test_c, '+-', color=mpnet_unsup_color, linewidth=1, label='mpNet Constrained Unsupervised')
plt.plot(model_sup.NMSE_mpnet_test_c, '-', color=mpnet_sup_color, linewidth=1, label='mpNet Constrained Supervised')


# upper bound
plt.plot(model.NMSE_mp_real, '-.', color=mp_real_color, linewidth=1.5, label='MP (Real dictionary)')

plt.grid()
plt.legend(loc='upper right')
plt.xlabel(r'Number of seen channels $(10^3)$',fontsize=16)
plt.ylabel('NMSE',fontsize=14)
# plt.title('NMSE Evolution vs number of seen channels')
plt.xlim(left=0)
# plt.xlim(right=3000)
plt.ylim(0, 1)
# plt.savefig(f"nmse_{noise_var:.0e}_L_{L}_T_{T}.pdf", format="pdf", bbox_inches="tight")
plt.show()



#%% #####################################################################################
# ############################## save estimations #######################################
# #######################################################################################

from utils.save_channel_estimations import save_estimation_mpnet,save_estimation_MP_LMMSE
import utils.generate_steering as generate_steering

data_file=f'Data/channels_var_snr/{noise_var:.0e}/T_{T}'
# -------------------------------- mpNet -------------------------------------------------------------------

data_pred_mpnet=f'Channel_estimation/{noise_var:.0e}/L_{L}_T_{T}/mpnet_c_unsup'
model_=f'pretrained_mpnet_models/{noise_var:.0e}/mpnet_c_unsup_L_{L}_T_{T}.pth'
save_estimation_mpnet(model_, data_file, data_pred_mpnet, batch_size, k, T, L,noise_var)
data_pred_mpnet=f'Channel_estimation/{noise_var:.0e}/L_{L}_T_{T}/mpnet_sup'
model_=f'pretrained_mpnet_models/{noise_var:.0e}/mpnet_sup_L_{L}_T_{T}.pth'
save_estimation_mpnet(model_, data_file, data_pred_mpnet, batch_size, k, T, L,noise_var)

# -------------------------------- LMMSE / MP ---------------------------------------------------------------
#get real and nominal antenna pos
path_init=Path.cwd()/'.saved_data'
antenna_pos = np.load(path_init/'Data/antenna_position.npz')
nominal_ant_positions=antenna_pos['nominal_position']
real_ant_positions=antenna_pos['real_position']

######Dictionnary parameters######
#random Azimuth angles uniformly distributed between 0 and 2*pi
DoA= np.load(path_init/f'Data/DoA.npz')['DoA']
g_vec= np.ones(nb_BS_antenna)
lambda_ =  0.010706874
#--------------channel_estimations--------------

data_pred_MP_nominal=f'Channel_estimation/{noise_var:.0e}/L_{L}_T_{T}/nominal'
data_pred_MP_real=f'Channel_estimation/{noise_var:.0e}/L_{L}_T_{T}/real'
data_pred_LMMSE=f'Channel_estimation/{noise_var:.0e}/L_{L}_T_{T}/lmmse'
batch_size_est=300

# construct nominal dictionary 
dict_nominal=generate_steering.steering_vect_c(torch.tensor(nominal_ant_positions).type(torch.FloatTensor),
                                                       torch.tensor(DoA).type(torch.FloatTensor),
                                                       torch.tensor(g_vec),
                                                       lambda_)
# construct real dictionary 
dict_real   =generate_steering.steering_vect_c(torch.tensor(real_ant_positions).type(torch.FloatTensor),
                                                       torch.tensor(DoA).type(torch.FloatTensor),
                                                       torch.tensor(g_vec),
                                                       lambda_)


save_estimation_MP_LMMSE(data_file, data_pred_MP_nominal, data_pred_MP_real, data_pred_LMMSE, batch_size, k, T, L, dict_nominal, dict_real,nb_BS_antenna,noise_var)


# %%
