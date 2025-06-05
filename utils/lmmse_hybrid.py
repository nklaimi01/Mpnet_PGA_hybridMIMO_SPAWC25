##%%
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##%% --------------- functions ------------------------------
def LMMSE(Y,M,M_tilde,C_h,sigma_2):
    '''
    sigma_2 : batched scalar of shape (N_u,)
    '''
    M_H=M.conj().transpose(-2,-1) # [*, A, L]
    M_tilde_H=M_tilde.conj().transpose(-2,-1) # [*, A, L]

    m1=torch.matmul(C_h.to(torch.complex128),M_H.to(torch.complex128)) # C_h*M_H shape [N_u, A, L] 
    m2=torch.matmul(M.to(torch.complex128),C_h.to(torch.complex128)) # M*C_h shape [N_u, L, A]
    m3=torch.matmul(m2.to(torch.complex128),M_H.to(torch.complex128)) # M*C_h*M_H shape [N_u, L, L]
    MM_tilde_H = torch.matmul(M_tilde, M_tilde_H) 
    C_n=sigma_2[:, None, None]*MM_tilde_H
    m4=m3+C_n #shape [N_u, L, L]
    C=torch.matmul(m1,torch.linalg.inv(m4)) # shape [N_u, A, L]
    # C: Shape [N_u, A, L]
    # Y: Shape [N_u, L]
    H_hat = torch.matmul(C.to(torch.complex128), Y.to(torch.complex128).unsqueeze(-1))  # shape [N_u, A]
    return H_hat

def LMMSE_estimation(h_clean,h_noisy,sigma_2,M,M_tilde):
    '''
    h_clean expected shape: [U, A]
    h_noisy expected shape: [U, AT]
    M shape: [A, TL]
    M_tilde shape: [AT, TL]
    '''
    # full_matrices=False pour la SVD
    # N_u=h_clean.shape[0] #number of channels 
    A=h_clean.shape[1] #size of channel (number of antennas)

    # LMMSE with batched matrices batch size B = N_u
    #batch should always be on dim=0
    M=M.mT #[*, TL, A]
    M_tilde=M_tilde.mT #[*, TL, AT]
    H_noisy=h_noisy.unsqueeze(2) #[*,192,1]

    H = h_clean.unsqueeze(2)     # [N_u, A, 1]
    H_H = h_clean.conj().unsqueeze(1)  # [N_u, 1, A]
    hh_H = torch.matmul(H,H_H)  # Shape will be [N_u, A, A]
    diagonals = torch.diagonal(hh_H, dim1=1, dim2=2)  # Shape: [N_u, A]
    C_h = torch.diag_embed(diagonals, dim1=1, dim2=2)  # Shape: [N_u, A, A]
    y=torch.matmul(M_tilde.to(torch.complex128),H_noisy.to(torch.complex128)).squeeze()
    H_hat=LMMSE(y,M,M_tilde,C_h,sigma_2)

    return H_hat.squeeze(0).squeeze(-1)
