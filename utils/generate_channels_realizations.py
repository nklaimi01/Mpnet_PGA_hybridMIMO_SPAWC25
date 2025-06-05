import numpy as np
import torch



def generate_noisy_channels_fixed_SNR(channels,snr_in_dB,T):# channels : shape [nb_channels,64]
    
    Nb_chan=channels.shape[0]
    Nb_BS_antenna=channels.shape[1]
    channels_norm=torch.zeros(Nb_chan)
    snr_in_lin = 10**(0.1*snr_in_dB)
     
     
    h_noisy=torch.empty([Nb_chan,0])
    sigma_2 = torch.linalg.norm(channels,2,dim=1)**2/(Nb_BS_antenna*snr_in_lin) 
    for t in range(T):
        n = (torch.sqrt(sigma_2).unsqueeze(-1))*(torch.randn(Nb_chan,Nb_BS_antenna)+1j*torch.randn(Nb_chan,Nb_BS_antenna)) # [N_u,A]
        h_noisy=torch.cat((h_noisy, channels+n), 1) # [N_u, AT]         

    channels_norm = torch.linalg.norm(h_noisy,2,dim=1) 
    h_noisy = h_noisy/channels_norm[:,None]
    
    
        
   
    return h_noisy,sigma_2      
    
        

        
def generate_noisy_channels_varying_SNR(channels,noise_var,T):
    channels=torch.tensor(channels,dtype=torch.complex128)
    Nb_chan=channels.shape[0]
    Nb_BS_antenna=channels.shape[1]
    channels_norm=torch.zeros(Nb_chan)
    sigma_2 = torch.full([Nb_chan],noise_var) 
    h_noisy=torch.empty([Nb_chan,0])
    for t in range(T):
        n = (torch.sqrt(sigma_2).unsqueeze(-1))*(torch.randn(Nb_chan,Nb_BS_antenna)+1j*torch.randn(Nb_chan,Nb_BS_antenna)) # [N_u,A]
        h_noisy=torch.cat((h_noisy, channels+n), 1) # [N_u, AT] 
     
        
        
    #epsilon = 1e-8

    #channels_norm = np.linalg.norm(h_noisy,2,axis=1) 
    #h_noisy = h_noisy/channels_norm[:,None]
        
   
    return torch.tensor(h_noisy,dtype=torch.complex128),torch.tensor(sigma_2)        
       
    
    
    