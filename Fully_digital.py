#%%
'''this scripts computes achievable sum-rate for a fully digital system using the water filling algorithm'''
import numpy as np
from pathlib import Path
import torch
import os

U     = 4    # Num of users
L     = 16   # RF chains
T     = 1   # Instant
A     = 64   # BS antennas
noise_var  = 2e-3 # noise variance
# noise_var_DL=A*noise_var
noise_var_DL=noise_var
test_size  = 1000 # NUMBER OF MIMO CHANNELS  


#Load data
path_init=Path.cwd()/'.saved_data'
dataset_dir = f'Data/{noise_var:.0e}/data_var_snr/T_{T}/test_data.npz'  
data = np.load(path_init/dataset_dir)

h=torch.tensor(data['h'],dtype=torch.complex128)

H_test=(h).view(-1,U,A)
P=np.linalg.norm(H_test,ord='fro',axis=(1,2))**2 / (A*noise_var_DL)
Pi=np.linalg.norm(h,2,axis=1)**2/ (A * noise_var_DL)
SNR_dB= 10*np.log10(P)


S1=[]
S2=[]

for i in range(H_test.shape[0]):

    # SVD of the channels
    u, lambda_, vh= torch.linalg.svd(H_test, full_matrices=True)
    g=np.abs(lambda_**2) # channel's gains^2
    g=g[i].unsqueeze(1).numpy()
    P=1  # if s2 formula will be used 
   #P=P # if s1 formula will be used


    # Bisection search for alpha 
    alpha_low = min(noise_var_DL/g) # Initial low
    alpha_high = (P + np.sum(noise_var_DL/g))/U # Initial high

    #print(alpha_high)
    #print(alpha_low)


    stop_threshold = 1e-15# Stop threshold


    # Iterate while low/high bounds are further than stop_threshold
    while(np.abs(alpha_low - alpha_high) > stop_threshold):

        alpha = (alpha_low + alpha_high) / 2 # Test value in the middle of low/high


        # Solve the power allocation
        p = alpha - noise_var_DL/g 
     
    
        p[p < 0] = 0 # Consider only positive power allocation
        

     
        if (np.sum(p) > P): # Exceeds power limit => lower the upper bound
            #print('upper')
            alpha_high = alpha
        else: # Less than power limit => increase the lower bound

            #print('low')
            alpha_low = alpha


    # Precoder Normalization check 
    F_BB=vh[i,:,:U].squeeze()
    
    F=F_BB @ np.diag(p.squeeze())

    # print(torch.linalg.norm(F,ord='fro')**2)


    #direct sum rate formula 
    h=H_test[i]

    Fopt=F
    a1 = torch.transpose(Fopt, 0, 1).conj() @ torch.transpose(h, 0, 1).conj()
    a3 = h @ Fopt @ a1
    a4 = torch.eye(U).reshape((U, U)) + a3/(U*noise_var_DL)  

    
    #s1 = torch.abs(torch.log(a4.det()))

    #Xater filling formula 
    s2= np.sum(np.log(1 + g*p/noise_var_DL))



    #S1.append(s1)
    S2.append(s2)


#print(np.mean(S1))
print(np.mean(S2))


# save sum rate
sum_rate={}
sum_rate['Fully_digital']=np.mean(S2)
save_dir=path_init /'sumRate'/f'{noise_var_DL:.0e}/L_{L}_T_{T}'
os.makedirs(save_dir,exist_ok=True)
np.savez(save_dir/'Fully_digital.npz', **sum_rate)


print('sum rate for fully digital system= ', sum_rate)
