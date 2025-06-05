from __future__ import print_function
from typing import Optional, Tuple
import numpy as np
import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.autograd.function import Function, FunctionCtx
import math
import utils.generate_steering as generate_steering

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom autograd Functions
class keep_k_max(Function):
   
    @staticmethod
    def forward(ctx: FunctionCtx, activations_in: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input = activations_in.clone().detach()
        
        if input.dim() == 1:
            input = input.unsqueeze(0)
        
        n_samples = input.shape[0]
        d_a = input.shape[1]

        # create activations out 
        activations_out = torch.zeros_like(input, dtype=torch.complex128)
        
        # find the top-k elements
        abs_input = torch.abs(input)
        _, topk_indices = torch.topk(abs_input, k, dim=1, largest=True, sorted=False)
        
        for i in range(n_samples):
            activations_out[i, topk_indices[i]] = input[i, topk_indices[i]]
        
        # Save activations that correspond to the selected atoms for backward propagation

        ctx.save_for_backward(activations_out)
        return activations_out, topk_indices
 
    @staticmethod
    def backward(ctx: FunctionCtx,grad_output: Tensor, id_k_max: Tensor) -> Tuple[Tensor, None, None]:
       
 
        activations_out, = ctx.saved_tensors
       
        grad_input = grad_output.clone()
        grad_input[activations_out == 0] = 0
        return grad_input, None, None
   

class mpNet_Constrained(nn.Module):
    def __init__(self, ant_position: torch.float, DoA: Tensor, g_vec: np.ndarray, lambda_, normalize: bool = True) -> None:
        super().__init__()
        
        
        self.ant_position = nn.Parameter((ant_position).to(device))
        self.DoA = DoA.to(device)
        self.g_vec = g_vec
        self.normalize = normalize
        self._lambda_ = lambda_
    
    def forward(self, x_M: Tensor, x: Tensor, M: Tensor, L: int, T:int, k: int, sigma: Optional[float] = None, sc: int = 2) -> Tuple[Tensor, Tensor, Optional[np.ndarray]]:
        residual_M = x_M.clone()
        residual = x.clone()

        if self.normalize:
            W = generate_steering.steering_vect_c(self.ant_position, self.DoA, self.g_vec, self._lambda_).to(device).type(torch.complex128)
            D_M = torch.matmul(torch.conj(M.mT), W).type(torch.complex128).to(device)
            # for i in range(D_M.shape[0]):
            #     norm = torch.tensor(np.sqrt(np.sum(np.abs(D_M[i].detach().cpu().numpy()) ** 2, 0))).to(device)
            #     D_M[i] = (D_M[i] / norm).type(torch.complex128).to(device)
            norm = torch.linalg.norm(D_M, dim=1, keepdim=True)  
            D_M = (D_M / norm).type(torch.complex128)  
        else:
            W = generate_steering.steering_vect_c(self.ant_position, self.DoA, self.g_vec, self._lambda_).type(torch.complex128).to(device)
            D_M = torch.matmul(torch.conj(M.mT), W).type(torch.complex128).to(device)
  
        D = W.clone()
        D=D.repeat(T,1)


        if sigma is None:  # No stopping criterion
            residuals = []
            for iter in range(k):
                z, id_k_max = keep_k_max.apply(residual_M @ torch.conj(D_M), 1)
                residual_M = residual_M - (z @ D_M.T)
                residuals.append(residual_M)

            h_hat_M = x_M - residual_M
            return residual_M, h_hat_M, None
        
       


        else:  # Use stopping criterion
            m = np.shape(residual_M)[1] # mesures m=T*L
            A = np.shape(residual)[1] # nb BS antenna
            if sc == 1:  # SC1
                threshold = pow(sigma, 2) * ( A*L + 2 * math.sqrt(A*L * math.log(A*L)))
            elif sc == 2:  # SC2
                threshold = pow(sigma, 2) * A * L 

            current_ids = list(range(residual_M.size()[0]))
      
            depths = np.zeros(residual_M.size()[0])
            iter = 0

            while bool(current_ids) and iter < 20:
                res_norm_2 = torch.norm(residual_M, p=2, dim=1) ** 2
                # print("residual shape",residual.shape):torch.Size([1000, 10])
                # print("M_D",M_D.shape):([1, 10, 1200])
                for i in current_ids[:]:
                    
                    if res_norm_2[i] < threshold[i]:
                        depths[i] = iter
                        current_ids.remove(i)
                        
                    else:
                 
                        # X_hat estimation [x= h @ M]

                        #initial code was (residual[i].clone() @ torch.conj(M_D[i]))  but index of M_D was out of bound 
                        if D_M.shape[0]==1:
                            z, id_k_max = keep_k_max.apply(residual_M[i].clone() @ torch.conj(D_M[0]), 1)
                            residual_M[i] = residual_M[i].clone()- (z @ D_M[0].mT).clone()

                        else:
                            z, id_k_max = keep_k_max.apply(residual_M[i].clone() @ torch.conj(D_M[i]), 1)
                            residual_M[i] = residual_M[i].clone()- (z @ D_M[i].mT).clone()

                        
                        # X_hat_m estimation [ x = h ]
                        residual[i] = residual[i] - (z @ D.T)
                

                iter += 1

            h_hat_M = x_M - residual_M
            h_hat = x - residual

            # print(iter)

        


            return residual_M, h_hat_M, h_hat
   

class mpNet(nn.Module):
   
   
       
    def __init__(self, W_init: Tensor) -> None:
            # W shape: [N,A]
            super().__init__()
 
                        
            self.W = nn.Parameter(W_init).to(device)
 
               
    def forward(self, x_M: Tensor,x: Tensor,M:Tensor, L:int,T:int,  k: int,sigma: Optional[float] = None, sc: int = 2) -> Tuple[Tensor, Tensor, Optional[np.ndarray]]:
            #X shape: (nb_samples,N,nb_users)
           
        residual_M = x_M.clone() #[TL]
        residual = x.clone() #[AT]
       
 
        ##M@ D
        D_M=torch.matmul(torch.conj(M.mT),self.W).to(device).type(torch.complex128)

        norm = torch.linalg.norm(D_M, dim=1, keepdim=True)  
        D_M = (D_M / norm).type(torch.complex128)  
 
 
        ##temprary dict to store chosen index
        D=self.W.clone()
        D=D.repeat(T,1)
 
        if sigma is None:  # no stopping criterion
            residuals = []
            for iter in range(k):
                z, id_k_max = keep_k_max.apply(residual_M @ torch.conj(self.W), 1)
 
                residual_M = residual_M -  (z  @   self.W.T)
     
                residuals.append(residual_M)
               
 
            x_hat_M = x_M - residual_M
           
 
            return residual_M,x_hat_M,None
 
 
 
        else:  # Use stopping criterion
            A = np.shape(residual)[1] # BS antenna
            if sc == 1:  # SC1
                threshold = pow(sigma, 2) * ( A*L + 2 * math.sqrt(A*L * math.log(A*L)))
            elif sc == 2:  # SC2
                threshold = pow(sigma, 2) * A * L 
            

            current_ids = list(range(residual_M.size()[0]))
      
            depths = np.zeros(residual_M.size()[0])
            iter = 0

            while bool(current_ids) and iter < 20:

                res_norm_2 = torch.norm(residual_M, p=2, dim=1) ** 2
         

                for i in current_ids[:]:
                    
                    if res_norm_2[i] < threshold[i]:
                        depths[i] = iter
                        current_ids.remove(i)
                    else:
                        # X_hat_M estimation [x= h @ M]
                        if D_M.shape[0]==1:
                            z, id_k_max = keep_k_max.apply(residual_M[i].clone() @ torch.conj(D_M[0]), 1)
                            residual_M[i] = residual_M[i].clone()- (z @ D_M[0].mT).clone()

                        else:
                            z, id_k_max = keep_k_max.apply(residual_M[i].clone() @ torch.conj(D_M[i]), 1)
                            residual_M[i] = residual_M[i].clone()- (z @ D_M[i].mT).clone()
                   

                        
                        # X_hat estimation [ x = h ]
                        residual[i] = residual[i] - (z @ D.T)
                

                iter += 1

            x_hat_M = x_M - residual_M
            x_hat = x - residual

            # print(iter)


            return residual_M, x_hat_M, x_hat