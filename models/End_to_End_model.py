import torch.nn as nn
import torch
seed_value=42
import numpy as np
import models.mpnet_model
import utils.generate_steering

# Define CompositeModel
class CompositeModel(nn.Module):
    
    def __init__(self, pga, mpnet,antenna_position,DoA,g_vec,lambda_):

        # w_init=generate_steering.steering_vect_c(torch.tensor(antenna_position).type(torch.FloatTensor),
        #                                                torch.tensor(DoA).type(torch.FloatTensor),
        #                                                torch.tensor(g_vec),
        #                                                lambda_).type(torch.complex128)
        super(CompositeModel, self).__init__()
        self.mpnet = mpnet
        self.pga = pga
        
    def forward(self, y,h_noisy,h_true,norm,k,sigma,M, U,A,L,T,num_of_iter,noise_var_DL):
        torch.manual_seed(seed_value)
         
        '''input--------------------------------------------------
       x= noisy channels multiplied by the measurement matrix
       h= noisy channels
       h_true = true channels
       k= max iteration of iterative mpnet 
       M= measurement matrix
       U=number of UE
       A=BS antenna
       L=RF chains
       num_of_iter=number of iteration for PGA
       noise_var_DL= the downlink noise variance that A times greater than the uplink one
       -------------------------------------------------------'''
        _ , _, h_hat= self.mpnet(y,h_noisy,M,L,T,k,sigma,2)
        '''expected norm : norm of the observed signal of size TL'''
        h_hat=h_hat[:,:A]
        h_hat_denorm=h_hat*(norm[:,None]/np.sqrt(T*L))
        sum_rate , wa, wd,WA,WD = self.pga(((h_hat_denorm).view(-1,U,A)).type(torch.complex128) ,U,L,num_of_iter,noise_var_DL)
        
        return sum_rate , wa, wd,WA,WD,h_hat_denorm
    


    def freeze_parameters(self,model):
        for param in model.parameters():
            param.requires_grad=False


    def unfreeze_parameters(self,model):
        for param in model.parameters():
            param.requires_grad=True

