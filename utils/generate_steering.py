import numpy as np
from typing import Tuple
import torch



def steering_vect_c(antenna_pos : np.ndarray, DoA : np.ndarray, g_vec : np.ndarray, lambda_ : float ) \
    -> Tuple[np.ndarray]:
    
        N=np.size(antenna_pos,0) #Number of antenna in the BS
        A=np.size(DoA,0) #Number of Atoms

        dict_=torch.zeros((N,A),dtype=torch.complex128)
            
        expo=torch.exp(-1j * 2 * np.pi * (1 / lambda_) * torch.matmul(antenna_pos, DoA.T))
                    
        dict_ = g_vec[:, None] * expo

            
        return dict_/torch.tensor(np.sqrt(np.sum(np.abs(dict_.detach().numpy())**2,0)))



    
    
    