#%%
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import matplotlib.patches as mpatches
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import models.uPGA_model
import models.End_to_End_model

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_init=Path.cwd().parent/'.saved_data'
L=10
U=4
T=1

# %% #############################################################################################
################## plot evaluationg using different channel estimation methods ###################
##################------------- SUM RATE vs PGA iterations --------------------###################
##################################################################################################

#functions:
def sum_loss(wa, wd, h, U, batch_size, noise_var_DL):
    a1 = torch.transpose(wa, 1, 2).conj() @ torch.transpose(h, 1, 2).conj()
    a2 = torch.transpose(wd, 1, 2).conj() @ a1
    a3 = h @ wa @ wd @ a2
    g = torch.eye(U,device=device).reshape((1, U, U)) + a3 / (U * noise_var_DL)  # g = Ik + H*Wa*Wd*Wd^(H)*Wa^(H)*H^(H)
    s = torch.log(g.det())  # s = log(det(g))

  
    loss = sum(torch.abs(s)) / batch_size
    return -loss

def evaluate_sum_rate(h, WA, WD, U, noise_var_DL, batch_size, num_iter):
    sum_rate = torch.zeros(num_iter+1)
    for i in range(num_iter+1):
        sum_rate[i] = sum_loss(WA[i], WD[i], h, U, batch_size, noise_var_DL)
           
    return -sum_rate

#parameters:
noise_var=2e-3 # uplink noise variance
L=16
T=1
U=4
A=64
noise_var_DL=noise_var # downlink noise variance

print(f'Uplink noise variance={noise_var:.0e}, Downlink noise variance={noise_var_DL:.0e}, L={L}, T={T}')
num_of_iter_pga_unf=10

#-------------------- collecting data -----------------------------------------------------
pga_models_dir=path_init/f'pretrained_upga_models/{noise_var_DL:.0e}'
E2E_models_dir=path_init/f'pretrained_E2E_models/{noise_var_DL:.0e}'
dataset_dir = f'Data/channels_var_snr/{noise_var:.0e}/T_{T}' 

test_data = np.load(path_init / dataset_dir/ f'test_data.npz')
M_test_data=np.load(path_init/'Data'/f'Measurement_matrix/L_{L}_T_{T}'/'test.npz')
h_test=torch.tensor(test_data['h'],dtype=torch.complex128)
H_true_test     = h_test.view(-1,U,A).to(device)
h_noisy_test=torch.tensor(test_data['h_noisy'],dtype=torch.complex128)
M_test = torch.tensor(M_test_data['M_test'],dtype=torch.complex128).to(device)
M_tilde_test = torch.tensor(M_test_data['M_tilde_test'],dtype=torch.complex128).to(device)
#observed signal:
y_test  =torch.matmul((h_noisy_test).unsqueeze(1),torch.conj(M_tilde_test)).squeeze().to(device)
denorm_factor_test  = torch.norm(y_test,p=2,dim=1)[:, None]/np.sqrt(T*L)
norm_y_test = torch.norm(y_test, p=2, dim=1)
#normalize signals 
y_test_normalized = y_test / norm_y_test[:, None]
h_noisy_test_normalized   = h_noisy_test / (norm_y_test[:,None]/np.sqrt(L))
h_test_normalized   = h_test / (norm_y_test[:,None]/np.sqrt(T*L))
sigma_2_test  =    torch.tensor(test_data['sigma_2'])   
sigma_2_test_normalized = torch.sqrt(sigma_2_test) / norm_y_test

#------------------- plotting figure ---------------------------------------------------------
plt.figure()
#L by L true channel + uPGA
model=torch.load(pga_models_dir/f'pga_H_true_L_{L}_T_{T}.pth')
s,_,_,WA,WD = model.forward(H_true_test, U, L,  num_of_iter_pga_unf,noise_var_DL)
sum_rate= evaluate_sum_rate(H_true_test, WA,WD,U,noise_var_DL,H_true_test.shape[0],num_of_iter_pga_unf)
y = sum_rate.cpu().detach().numpy()
x = np.arange(len(y)) 

#water filling (fully digital):
Fully_digital=np.load(path_init/'sumRate'/f'{noise_var_DL:.0e}/L_{L}_T_{T}/Fully_digital.npz')['Fully_digital']


#upga using true channel knowledge
color_true_h='tab:cyan'
plt.plot(x, y, '+-',color=color_true_h,label='True channels')


estimators=['mpnet_sup']
labels=['E2E warm start']
styles=['v-']
colors=['tab:olive']
for i,estimator in enumerate(estimators):
    model=torch.load(E2E_models_dir/f'E2E_{estimator}_L_{L}_T_{T}.pth')
    s,_,_,WA,WD,_ = model.forward(y_test_normalized,h_noisy_test_normalized,h_test_normalized,norm_y_test,8,sigma_2_test_normalized,M_test,U,A,L,T,num_of_iter_pga_unf,noise_var_DL)
    sum_rate=evaluate_sum_rate(H_true_test,WA,WD,U,noise_var_DL,H_true_test.shape[0],10)
    y = sum_rate.cpu().detach().numpy()
    plt.plot(x, y, styles[i],color=colors[i],label=labels[i])

estimators=['mpnet_sup','mpnet_c_unsup']
labels=['LbyL supervised','LbyL unsupervised']
styles=['^-','x-']
colors=['tab:blue','tab:orange']
for i,estimator in enumerate(estimators):
    estimation_dir = f'Channel_estimation/{noise_var:.0e}/L_{L}_T_{T}/{estimator}'
    est_test_data = np.load(path_init / estimation_dir / f'test.npz') 
    h_est = torch.tensor(est_test_data['channels'], dtype=torch.complex128)[:,:A]
    h_est     = h_est   * denorm_factor_test
    est_test_data = np.load(path_init / estimation_dir / f'test.npz') 
    H_est_test    = h_est.view(-1,U,A).to(device)
    model=torch.load(pga_models_dir/f'pga_{estimator}_L_{L}_T_{T}.pth')
    s,_,_,WA,WD = model.forward(H_est_test, U, L,  num_of_iter_pga_unf,noise_var_DL)
    sum_rate= evaluate_sum_rate(H_true_test, WA,WD,U,noise_var_DL,H_true_test.shape[0],num_of_iter_pga_unf)
    y = sum_rate.cpu().detach().numpy()
    x = np.arange(len(y)) 
    plt.plot(x, y, styles[i],color=colors[i],label=f'{labels[i]}')

estimators=['naive']
labels=['E2E cold start']
styles=['o-']
colors=['tab:pink']
for i,estimator in enumerate(estimators):
    model=torch.load(E2E_models_dir/f'E2E_{estimator}_L_{L}_T_{T}.pth')
    s,_,_,WA,WD,_ = model.forward(y_test_normalized,h_noisy_test_normalized,h_test_normalized,norm_y_test,8,sigma_2_test_normalized,M_test,U,A,L,T,num_of_iter_pga_unf,noise_var_DL)
    sum_rate=evaluate_sum_rate(H_true_test,WA,WD,U,noise_var_DL,H_true_test.shape[0],10)
    y = sum_rate.cpu().detach().numpy()
    plt.plot(x, y, styles[i],color=colors[i],label=labels[i])


estimators=['lmmse']
labels=['LMMSE']
styles=['s-']
colors=['tab:red']
for i,estimator in enumerate(estimators):
    estimation_dir = f'Channel_estimation/{noise_var:.0e}/L_{L}_T_{T}/{estimator}'
    est_test_data = np.load(path_init / estimation_dir / f'test.npz') 
    h_est = torch.tensor(est_test_data['channels'], dtype=torch.complex128)[:,:A]
    h_est     = h_est   * denorm_factor_test
    est_test_data = np.load(path_init / estimation_dir / f'test.npz') 
    H_est_test    = h_est.view(-1,U,A).to(device)
    model=torch.load(pga_models_dir/f'pga_{estimator}_L_{L}_T_{T}.pth')
    s,_,_,WA,WD = model.forward(H_est_test, U, L,  num_of_iter_pga_unf,noise_var_DL)
    sum_rate= evaluate_sum_rate(H_true_test, WA,WD,U,noise_var_DL,H_true_test.shape[0],num_of_iter_pga_unf)
    y = sum_rate.cpu().detach().numpy()
    x = np.arange(len(y)) 
    plt.plot(x, y, styles[i],color=colors[i],label=f'{labels[i]}')


# plt.title(f'Sum rate vs PGA iterations, A={A} U={U} L={L} T={T}')
plt.xlabel('Number of Iterations',fontsize=14)
plt.ylabel('Achievable Rate',fontsize=14)
plt.grid()

upper_bound_patch = mpatches.Patch(color='none', label=f"Fully digital = {Fully_digital:.2F}")

# Add legend with proxy artist
plt.legend(handles=[upper_bound_patch] + plt.gca().get_legend_handles_labels()[0],loc='lower right', bbox_to_anchor=(1, 0.2))

# plt.legend(loc='best')
plt.xlim(left=0)
plt.savefig("sumrate_10iter.pdf", format="pdf", bbox_inches="tight")
plt.show()


#%% ##############################################################################################
################## plot evaluationg using different channel estimation methods ###################
##################------------------- SUM RATE vs SNR--------------------------###################
##################################################################################################

noise_variances=[6e-04, 2e-03, 6e-03, 2e-02, 6e-02, 2e-01]
unf_mpnet_unsup_list,unf_mpnet_sup_list,E2E_cold_start_list,E2E_warm_start_list,unf_lmmse_list,uPGA_True_channels_list,Fully_digital_list=[],[],[],[],[],[],[]
for i,s in enumerate(noise_variances):
    path=path_init/'sumRate'/f'{s:.0e}/L_{L}_T_{T}'
    Fully_digital=np.load(path/'Fully_digital.npz')['Fully_digital']
    uPGA_True_channels= np.load(path/'uPGA_true_channels.npz')['uPGA_true_channels']
    E2E_warm_start=np.load(path/'E2E_warm_start.npz')['E2E_warm_start']
    unf_mpnet_sup= np.load(path/'unf_est_mpnet_sup.npz')['unf_est_mpnet_sup']
    unf_mpnet_c_unsup= np.load(path/'unf_est_mpnet_c_unsup.npz')['unf_est_mpnet_c_unsup']
    E2E_cold_start=np.load(path/'E2E_cold_start.npz')['E2E_cold_start']
    unf_lmmse=np.load(path/'unf_est_lmmse.npz')['unf_est_lmmse']

    Fully_digital_list.append(Fully_digital)
    uPGA_True_channels_list.append(uPGA_True_channels)
    E2E_warm_start_list.append(E2E_warm_start)
    unf_mpnet_sup_list.append(unf_mpnet_sup)
    unf_mpnet_unsup_list.append(unf_mpnet_c_unsup)
    E2E_cold_start_list.append(E2E_cold_start)
    unf_lmmse_list.append(unf_lmmse)


y1 = np.array([arr for arr in Fully_digital_list])
y2 = np.array([arr[-1] for arr in uPGA_True_channels_list])
y3 = np.array([arr[-1] for arr in E2E_warm_start_list])
y4 = np.array([arr[-1] for arr in unf_mpnet_sup_list])
y5 = np.array([arr[-1] for arr in unf_mpnet_unsup_list])
y6= np.array([arr[-1] for arr in E2E_cold_start_list])
y7 = np.array([arr[-1] for arr in unf_lmmse_list])

x=np.arange(len(y1))
x=noise_variances
plt.plot(x, y1, 'p-',label='Water filling')
plt.plot(x, y2, 'v-',label='Unfolded PGA(10 iter) - Perfect CSI')
plt.plot(x, y3, 'x--',label='End To End supervised')
plt.plot(x, y4, 'o-',label='Unfolded PGA(10 iter) - MpNet supervised Estimation')
plt.plot(x, y5, 'x--',label='Unfolded PGA(10 iter) - MpNet unsupervised Estimation ')
plt.plot(x, y6, 'x--',label='End To End naive')
plt.plot(x, y7, 'p-',label='Unfolded PGA(10 iter) - LMMSE Estimation')


# Personnalisation des ticks de l'axe des x
plt.xticks(noise_variances)
# plt.xlim(6e-4,1e-03) 
# Inversion de l'axe des x
plt.gca().invert_xaxis()


plt.title(f'L={L}, T={T}, U={U}')
plt.xlabel('Noise variance')
plt.ylabel('Achievable Rate')
plt.grid(True)
plt.legend(loc="upper right", bbox_to_anchor=(1, -0.1)) 

plt.show()
