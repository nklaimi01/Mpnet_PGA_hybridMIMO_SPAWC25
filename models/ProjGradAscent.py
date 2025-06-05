import torch
import torch.nn as nn
import numpy as np

seed_value=42
torch.manual_seed(seed_value)

class ProjGA(nn.Module):

    def __init__(self, mu):
        super().__init__()
        self.mu = nn.Parameter(mu)  # parameters = (mu_a, mu_(d,1), ..., mu_(d,B))
 
    def forward(self, h, U, L,  num_of_iter,noise_var_DL):
        torch.manual_seed(seed_value)

        '''------- Projection Gradient Ascent execution ---------
 
        --- inputs:
        h - channel realization
        n - num of users
        l1 - num of RF chains
        num_of_iter - num of iters of the PGA algorithm
 
        ---- initializing variables
        svd for H_avg --> H = u*smat*vh'''

        H=h.clone().detach()


        _, S, vh = torch.linalg.svd(H, full_matrices=True)
        # vh.requires_grad=False
        # initializing Wa as vh
   
        


        #print('vh',torch.abs(vh[:, :, :l1]))
        epsilon=1e-12
        # wa = vh[:, :, :L]/torch.abs(vh[:, :, :L]+epsilon)
        # Do NOT set requires_grad=False, keep it as it is
        wa = vh[:, :, :L]
        # wa=(torch.randn(h.shape[0],64 ,L )+1j*torch.randn(h.shape[0],64 ,L )).to(torch.complex128)
        wa = wa / (torch.abs(wa) + epsilon)  # Normalize phase

        # randomizing Wd
        wd = (torch.randn(h.shape[0],L ,U )+1j*torch.randn(h.shape[0],L ,U )).to(torch.complex128)
    
       
        # projecting Wd onto the power constraint
        wd = (torch.sqrt(U / torch.linalg.matrix_norm(wa @ wd, ord='fro')**2)).reshape(h.shape[0], 1, 1) * wd
 
        # defining an array which holds the values of the rate of each iteration
        obj = torch.zeros(num_of_iter+1, h.shape[0])
        W_A = torch.zeros(num_of_iter+1, wa.shape[0], wa.shape[1], wa.shape[2]).type(torch.complex128)
        W_D = torch.zeros(num_of_iter+1, wd.shape[0], wd.shape[1], wd.shape[2]).type(torch.complex128)
 

        W_A[0]=wa
        W_D[0]=wd   
        obj[0] = self.objec((np.sqrt(1 /(U*noise_var_DL)))*h, wa, wd, U)
        # update equations
        for x in range(num_of_iter):
                

                # ---------- Wa  ---------------
                # gradient ascent
                wa_t = wa+ self.mu[x][0] * self.grad_wa((np.sqrt(1 /(U*noise_var_DL)))*h, wa, wd, U)

              
                # projection of Wa onto the amplitude constraint
                wa=wa_t/torch.abs(wa_t+epsilon)
                
    
                # ---------- Wd ---------------
                
                # gradient ascent
                wd_t= wd + self.mu[x][ 1] * self.grad_wd((np.sqrt(1 /(U*noise_var_DL)))*h, wa, wd, U)
                # projection onto the power constraint
                wd = (torch.sqrt(U / torch.linalg.matrix_norm(wa @ wd_t, ord='fro')**2)).reshape(h.shape[0], 1, 1) * wd_t

                W_A[x+1]=wa
                W_D[x+1]=wd 
    
                # update the rate
                obj[x+1] = self.objec((np.sqrt(1 /(U*noise_var_DL)))*h, wa, wd, U)
                #print('obj[x]',obj[x])
    
        return torch.transpose(obj, 0, 1), wa, wd, W_A,W_D
 
    def objec(self, h, wa, wd, U):
        # calculates the rate for a given channel (h) and precoders (wa, wd)
        return torch.abs(torch.log((torch.eye(U).reshape((1,  U, U)) +(
                       h @ wa @ wd @ torch.transpose(wd, 1, 2).conj() @
                       torch.transpose(wa, 1, 2).conj() @ torch.transpose(h, 1, 2).conj())).det()))
 

 
    def grad_wa(self, h, wa, wd, U):
        # calculates the gradient with respect to wa for a given channel (h) and precoders (wa, wd)
        return( torch.transpose(h, 1, 2) @ torch.transpose(torch.linalg.inv(torch.eye(U).reshape((1,  U, U))
                                                                             + h @ wa @ wd @
                                                                             torch.transpose(wd, 1, 2).conj() @
                                                                             torch.transpose(wa, 1, 2).conj() @
                                                                             torch.transpose(h, 1, 2).conj()), 1, 2)
                                                                             @ h.conj() @ wa.conj() @ wd.conj() @
                                                                             torch.transpose(wd, 1, 2))
         
 
    def grad_wd(self, h, wa, wd, U):
        # calculates the gradient with respect to wd for a given channel (h) and precoders (wa, wd)
        return (torch.transpose(wa, 1, 2) @ torch.transpose(h, 1, 2) @
                torch.transpose(torch.linalg.inv(torch.eye(U).reshape((1, U, U)) +
                h @ wa @ wd @ torch.transpose(wd, 1, 2).conj() @
                torch.transpose(wa, 1, 2).conj() @ torch.transpose(h, 1, 2).conj()), 1, 2) @
                h.conj() @ wa.conj() @ wd.conj())