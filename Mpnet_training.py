from typing import  Tuple
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
import utils.generate_steering as generate_steering
import  models.mpnet_model as mpnet_model
import utils.sparse_recovery as sparse_recovery
import sys
import utils.utils_function as utils_function
from torch.nn.utils import parameters_to_vector
import matplotlib as mpl
from tqdm import trange
import os
from utils.lmmse_hybrid import LMMSE_estimation
import time
from tqdm import tqdm



mpl.rcParams.update(mpl.rcParamsDefault)
#torch.manual_seed(42)

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
 
class UnfoldingModel_Sim:
    def __init__(self,
                 BS_position: np.ndarray,
                 nominal_antenna_pos: np.ndarray,
                 real_antenna_pos: np.ndarray,
                 sigma_ant:float,
                 DoA: np.ndarray,
                 g_vec: np.ndarray,
                 lambda_,
                 snr_in_dB: int = 20,                  
                 lr: float = 0.001,
                 lr_constrained: float =0.1,
                 momentum: float = 0.9,                
                 optimizer = 'adam',
                 epochs: int = 1,
                 batch_size: int = 100,              
                 k: int = None,
                 batch_subsampling: int = 20,
                 train_type: str = 'online',
                 f0:int=28e9,
                 nb_channels_test: int=200,
                 nb_channels_train: int=1000,
                 nb_channels_val: int=200,
                 nb_channels_per_batch: int=20,
                 nb_BS_antenna: int=64,
                 L: int=10,
                 T: int=1,
                 noise_var: int=1e-02
                 ) -> None:
       
       
        self.snr_in_dB=snr_in_dB
        self.lr=lr
        self.lr_constrained=lr_constrained
        self.momentum=momentum
        self.epochs=epochs
        self.k=k
        self.batch_subsampling=batch_subsampling
        self.train_type=train_type
        self.BS_position=BS_position
        self.nominal_antenna_pos=torch.tensor(nominal_antenna_pos).type(torch.FloatTensor).to(device)
        self.DoA=torch.tensor(DoA).type(torch.FloatTensor).to(device)
        self.g_vec=torch.tensor(g_vec).to(device)
        self.lambda_= lambda_
        self.train_type=train_type
        self.f0=f0
        self.batch_number=batch_size
        self.nb_channels_test=nb_channels_test
        self.nb_channels_train= nb_channels_train
        self.nb_channels_val=nb_channels_val
        self.optimizer=optimizer
        self.nb_channels_per_batch=nb_channels_per_batch
        self.sigma_ant=sigma_ant
        self.real_antenna_pos=torch.tensor(real_antenna_pos).type(torch.FloatTensor).to(device)
        self.m=L*T
        self.T=T
        self.L=L
        self.noise_var=noise_var
        self.nb_BS_antenna = nb_BS_antenna
        self.times = None
        self.start_time = None
 
        # to store the channel realizations
        self.dataset=None
     
        #sigma noise
        self.sigma_noise = None
       
        #stopping criteria
        self.SC2 = None
 
 
        #generate nominal steering vectors using nominal antenna positions
        dict_nominal=generate_steering.steering_vect_c(self.nominal_antenna_pos,
                                                       self.DoA,
                                                       self.g_vec,
                                                       self.lambda_).type(torch.complex128).to(device)
        #
        dict_real = generate_steering.steering_vect_c(self.real_antenna_pos,
                                                       self.DoA,
                                                       self.g_vec,
                                                       self.lambda_).type(torch.complex128).to(device)
       
       
 
        self.dict_nominal = dict_nominal
        self.dict_real=dict_real
        weight_matrix=dict_nominal
       
        #initialization of the mpnet model with the weight matrix
        self.mpNet = mpnet_model.mpNet(weight_matrix).to(device)
        self.mpNet_Constrained= mpnet_model.mpNet_Constrained(self.nominal_antenna_pos,self.DoA,self.g_vec,self.lambda_, True).to(device)
       
        #Initialization of the optimizer
        self.optimizer= optim.Adam(self.mpNet.parameters(),lr=self.lr)
        self.constrained_optimizer = optim.Adam(self.mpNet_Constrained.parameters(), lr = self.lr_constrained)
 
       
        #Result table for every batch size over the whole epochs
        #cost function over train
        self.cost_func = np.zeros(epochs*batch_size)
        self.cost_func_c=np.zeros(epochs*batch_size)
       
       
       
        dim_results=int(np.ceil(epochs*batch_size/batch_subsampling))-epochs+1 #batch 0 when epochs>1 are not appended
       
        #cost function over test    
        self.NMSE_mpnet_test = np.zeros(dim_results)
        self.NMSE_mpnet_test_c = np.zeros(dim_results)
        self.NMSE_lmmse =np.zeros(dim_results)
        self.NMSE_mp_nominal=np.zeros(dim_results)
        self.NMSE_mp_real=np.zeros(dim_results)
       
        self.times=np.zeros(dim_results)

       
       
       
    def train_online_test_inference(self,noise_var,supervised=False):
        if supervised:
            print('supervised learning')
        else:
            print('unsupervised learning')

        if self.train_type=='Online':
 
            path_init=Path.cwd()/'.saved_data'
            file_name = f'Data/{noise_var:.0e}/data_var_snr/T_{self.T}/test_data.npz'  
            test_data = np.load(path_init/file_name)
            M_data=np.load(path_init/'Data'/f'Measurement_matrix/L_{self.L}_T_{self.T}'/'test.npz')
                
            h_test        =    torch.tensor(test_data['h'],dtype=torch.complex128).to(device)         
            h_noisy_test  =    torch.tensor(test_data['h_noisy'],dtype=torch.complex128).to(device)       
            sigma_2_test  =    torch.tensor(test_data['sigma_2']).to(device) 
            M_test = torch.tensor(M_data['M_test'],dtype=torch.complex128).to(device)
            M_tilde_test = torch.tensor(M_data['M_tilde_test'],dtype=torch.complex128).to(device)  

            #the observed signal:
            y_test =torch.matmul((h_noisy_test).unsqueeze(1),torch.conj(M_tilde_test)).squeeze().to(device)


            ## preprocessing 
            # norm_noisy_test     =    torch.norm(h_noisy_test,p=2,dim=1).to(device) #noisy channel's norm
            norm_y_test     =torch.norm(y_test,p=2,dim=1).to(device)
            # normalize channels 
            y_test  =y_test/ norm_y_test[:,None]
            # h_noisy_test   = h_noisy_test / norm_noisy_test[:,None]
            h_noisy_test   = h_noisy_test / (norm_y_test[:,None]/np.sqrt(self.L))
            h_test         = h_test / (norm_y_test[:,None]/np.sqrt(self.T*self.L))
            # h_test         = h_test / (norm_noisy_test[:,None]/np.sqrt(self.T))
            
            #Stopping criteria
            sigma_2_test = (torch.sqrt(sigma_2_test)/norm_y_test).to(device) #qst: threshold in mpnet model is camculated using this
            SC2= pow((torch.sqrt(sigma_2_test)),2) * self.nb_BS_antenna

            idx_subs = 1
            for i in tqdm(range(self.epochs)):
                for batch in range(self.batch_number):
                    #############################################
                    ########## test for batch 0 #################
                    #############################################  
                    if batch == 0 and i==0:
                    #the first inference before training the channel to get estimation with the initial nominal dictionnary(no gradient is computed)
                        self.start_time=time.time()
                    
                        with torch.no_grad():  
                            residuals_test_M, est_chan_test_M, est_chan_test = self.mpNet(y_test,h_noisy_test,M_test,self.L,self.T, self.k,sigma_2_test,2)
                            residuals_test_M_c, est_chan_test_M_c, est_chan_test_c= self.mpNet_Constrained(y_test,h_noisy_test,M_test,self.L,self.T, self.k,sigma_2_test,2)
                    
                        #Estimate channel reconstruction
                        h_hat_mpNet_test = est_chan_test.detach().cpu().numpy()
                        h_hat_mpNet_test_c = est_chan_test_c.detach().cpu().numpy()
     
                        #Cost function  
                        self.NMSE_mpnet_test[0]=torch.mean(torch.sum(torch.abs(h_test.cpu()-h_hat_mpNet_test[:,:self.nb_BS_antenna])**2,1)/torch.sum(torch.abs(h_test.cpu())**2,1))
                        self.NMSE_mpnet_test_c[0]=torch.mean(torch.sum(torch.abs(h_test.cpu()-h_hat_mpNet_test_c[:,:self.nb_BS_antenna])**2,1)/torch.sum(torch.abs(h_test.cpu())**2,1))
        

                        #MP with real and nominal dictionnary
                        nmse_mp_nominal, _= UnfoldingModel_Sim.run_mp_omp(self,h_test,y_test,h_noisy_test,M_test,SC2,'nominal')
                        nmse_mp_real,_    = UnfoldingModel_Sim.run_mp_omp(self,h_test,y_test,h_noisy_test,M_test,SC2,'real')
                    
        
                    
                        #Add mp results to tables
                        self.NMSE_mp_nominal[:] = nmse_mp_nominal * np.ones(len(self.NMSE_mp_nominal))
                    
                        self.NMSE_mp_real[:] = nmse_mp_real * np.ones(len(self.NMSE_mp_real))
                    
                        ##################### LMMSE estimation ###################
                    
                        lmmse_est=LMMSE_estimation(h_test,h_noisy_test,sigma_2_test,M_test,M_tilde_test)
                        cost_func_lmmse=np.mean(np.linalg.norm(h_test.cpu().numpy()-lmmse_est.cpu().numpy(),2,axis=1)**2/np.linalg.norm(h_test.cpu().numpy(),2,axis=1)**2)
                        self.NMSE_lmmse[:]=cost_func_lmmse*np.ones(len(self.NMSE_lmmse))

                        print(f'batch {batch} MP Nominal: {self.NMSE_mp_nominal[0]} ')
                        print(f'batch {batch} MP Real: {self.NMSE_mp_real[0]} ')
                        print(f'batch {batch} LMMSE: {self.NMSE_lmmse[0]}')
                        print(f'batch {batch} MpNet: {self.NMSE_mpnet_test[0]}')
                        print(f'batch {batch} Constrained MpNet: {self.NMSE_mpnet_test_c[0]} ')
                    
                    #############################################
                    ################ training ###################
                    #############################################                
                    
                    # Load Train channel
                    path_init=Path.cwd()/'.saved_data'
                    file_name = f'Data/{noise_var:.0e}/data_var_snr/T_{self.T}/batch_{batch}.npz'  
                    train_data = np.load(path_init/file_name)
                    M_data=np.load(path_init/'Data'/f'Measurement_matrix/L_{self.L}_T_{self.T}'/f'batch_{batch}.npz')
                    
                    h_train        =    torch.tensor(train_data['h'],dtype=torch.complex128).to(device)         
                    h_noisy_train  =    torch.tensor(train_data['h_noisy'],dtype=torch.complex128).to(device)       
                    sigma_2_train  =    torch.tensor(train_data['sigma_2']).to(device)      
                    M_train = torch.tensor(M_data['M_train'],dtype=torch.complex128).to(device)
                    M_tilde_train = torch.tensor(M_data['M_tilde_train'],dtype=torch.complex128).to(device)

                    #the observed signal:
                    y_train  =torch.matmul((h_noisy_train).unsqueeze(1),torch.conj(M_tilde_train)).squeeze().to(device)

                    ##preprocessing

                    # normalize channels
                    norm_y_train     = torch.norm(y_train,p=2,dim=1).to(device)
                    norm_noisy_train     =    torch.norm(h_noisy_train,p=2,dim=1).to(device)
                    y_train  = y_train/ norm_y_train[:,None]
                    # h_noisy_train   = h_noisy_train / norm_noisy_train[:,None]
                    h_noisy_train   = h_noisy_train / (norm_y_train[:,None]/np.sqrt(self.L))
                    # h_train         = h_train / (norm_noisy_train[:,None]/np.sqrt(self.T))
                    h_train         = h_train / (norm_y_train[:,None]/np.sqrt(self.T*self.L))
                    #Stopping criteria
                    self.sigma_noise = (torch.sqrt(sigma_2_train)/norm_y_train).to(device)

                    ###########################################################################
                    ##########################  forward propagation  ##########################
                    ###########################################################################
                    # Reset gradients to zero
                    self.optimizer.zero_grad()
                    self.constrained_optimizer.zero_grad()
                    residuals_M, est_chan_M, est_chan= self.mpNet(y_train,h_noisy_train,M_train,self.L,self.T, self.k,self.sigma_noise,2)
                    residuals_M_c, est_chan_M_c, est_chan_c= self.mpNet_Constrained(y_train,h_noisy_train,M_train,self.L,self.T,  self.k,self.sigma_noise,2)
                    

                    ###########################################################################
                    ##########################  Backward propagation  #########################
                    ###########################################################################
                    if not supervised:
                        # unsupervised version:
                        out_mp = torch.abs(residuals_M).pow(2).sum() / h_noisy_train.shape[0] #add H*M for T>1
                        out_mp_c = torch.abs(residuals_M_c).pow(2).sum() / h_noisy_train.shape[0]
                    else:
                        # supervised version:
                        out_mp= torch.abs(h_train - est_chan[:,:self.nb_BS_antenna]).pow(2).sum() / h_train.shape[0]
                        out_mp_c= torch.abs(h_train - est_chan_c[:,:self.nb_BS_antenna]).pow(2).sum() / h_train.shape[0]
                    

                    out_mp.backward()      
                    out_mp_c.backward()
                
                    #Gradient calculation
                    self.optimizer.step()
                    self.constrained_optimizer.step()

        
                    if batch == self.batch_number-1:
                        #----save the model----
                        path_init=Path.cwd()/'.saved_data'
                        save_dir=path_init /f'pretrained_mpnet_models/{self.noise_var:.0e}'
                        os.makedirs(save_dir, exist_ok=True)  # Creates folder if it doesn't exist
                        sup_unsup = "sup" if supervised else "unsup"
                        torch.save(self.mpNet_Constrained,save_dir/f'mpnet_c_{sup_unsup}_L_{self.L}_T_{self.T}.pth')
                        torch.save(self.mpNet,save_dir/f'mpnet_{sup_unsup}_L_{self.L}_T_{self.T}.pth')

                
                    #############################################
                    ###### testing after batch subsampling ######
                    #############################################
                    if batch%self.batch_subsampling == 0 and batch != 0:
                        self.times[idx_subs] = time.time() - self.start_time
                        with torch.no_grad():  
            
                            residuals_test_M, est_chan_test_M, est_chan_test = self.mpNet(y_test,h_noisy_test,M_test,self.L, self.T, self.k,sigma_2_test,2)
                            residuals_test_M_c, est_chan_test_M_c, est_chan_test_c= self.mpNet_Constrained(y_test,h_noisy_test,M_test,self.L, self.T, self.k,sigma_2_test,2)
                    
                        #Estimate channel reconstruction
            
                        h_hat_mpNet_test =  est_chan_test.detach().numpy()
                        h_hat_mpNet_test_c = est_chan_test_c.detach().cpu().numpy()
        
                    
        
                        #Cost function  
                        self.NMSE_mpnet_test[idx_subs]=torch.mean(torch.sum(torch.abs(h_test.cpu()-h_hat_mpNet_test[:,:self.nb_BS_antenna])**2,1)/torch.sum(torch.abs(h_test.cpu())**2,1))
                        self.NMSE_mpnet_test_c[idx_subs]=torch.mean(torch.sum(torch.abs(h_test.cpu()-h_hat_mpNet_test_c[:,:self.nb_BS_antenna])**2,1)/torch.sum(torch.abs(h_test.cpu())**2,1))
        
                    
                        print(f'batch {batch} MP Nominal: {self.NMSE_mp_nominal[idx_subs]} ')
                        print(f'batch {batch} MP Real: {self.NMSE_mp_real[idx_subs]} ')
                        print(f'batch {batch} LMMSE: {self.NMSE_lmmse[idx_subs]}')
                        print(f'batch {batch} MpNet: {self.NMSE_mpnet_test[idx_subs]}')
                        print(f'batch {batch} Constrained MpNet: {self.NMSE_mpnet_test_c[idx_subs]} ')
        
        
                        idx_subs += 1          


        elif self.train_type=='Offline': 
            print("offline training option not implemented")
                                  

                        
                       
                       

                       
                        
    def run_mp_omp(self,h:np.ndarray, h_noisy_M:np.ndarray,h_noisy:np.ndarray,M:np.ndarray, SC: np.ndarray, type_dict: str) -> Tuple[float,float]:
           
 
            out_chans= torch.zeros_like(h,dtype=torch.complex128)
   
 
            for i in range(h_noisy_M.shape[0]):
                if type_dict == 'nominal':
                    
 
                    dict_nominal_M=torch.matmul(torch.conj(M.mT) , self.dict_nominal).type(torch.complex128).to(device)
                    dict_nominal_M=dict_nominal_M.squeeze(0)

                    out_chans[i,:]= sparse_recovery.mp(h_noisy[i,:],h_noisy_M[i,:],self.dict_nominal,dict_nominal_M,self.k,False,SC[i])
               
                elif type_dict == 'real':
                    dict_real_M=torch.matmul(torch.conj(M.mT) , self.dict_real).type(torch.complex128).to(device)
                    dict_real_M=dict_real_M.squeeze(0)
                     
                    out_chans[i,:]= sparse_recovery.mp(h_noisy[i,:],h_noisy_M[i,:],self.dict_real,dict_real_M,self.k,False,SC[i])
           
                else:
               
                    sys.exit('undefined dictionnary type, either real or nominal!')
               
            rel_err=np.mean(np.linalg.norm(out_chans.cpu()-h.cpu(),2,axis=1)**2/np.linalg.norm(h.cpu(),2,axis=1)**2)
       
            snr_out_db = 10*np.log10(1/rel_err)
           
       
            return rel_err,snr_out_db
                 
       
 
