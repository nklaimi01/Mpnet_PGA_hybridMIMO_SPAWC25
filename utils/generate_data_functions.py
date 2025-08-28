import numpy as np 
from utils.utils_function import init_scene
from sionna.rt import Transmitter
import sionna
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tqdm import tqdm
import torch

#generate  dataset of channels using sionna ray tracing library
def generate_dataset(save_dir,nb_batches,size_batches,nb_BS_antennas):
    os.makedirs(save_dir,exist_ok=True)
    f0=28e9 #HZ 
    BS_position=[100, -90, 30]
    scene,nominal_ant_positions,real_ant_positions =init_scene(BS_position,f0)

    scene.preview()

    #Save nominal and real positions
    antenna_pos={}
    antenna_pos['nominal_position']=nominal_ant_positions
    antenna_pos['real_position']=real_ant_positions
    file_name = "antenna_position.npz"  
    np.savez( save_dir/ file_name, **antenna_pos)


    #Generate UE positions 
    nb_UE_scene=10000000
    cent_loc =[100,20,25]

    dx=2
    dy=2
    num_positions=np.sqrt(nb_UE_scene)

    x_arr = np.arange(cent_loc[0] - num_positions  / 2, cent_loc[0] + num_positions  / 2, dx)
    y_arr = np.arange(cent_loc[1] - num_positions  / 2, cent_loc[1] + num_positions  / 2, dy)

    xx, yy = np.meshgrid(x_arr, y_arr)
        
    #generate grid positions
    locs_grid = np.stack(( xx.flatten(),yy.flatten(), np.full(xx.size,cent_loc[2]))).T


    #generate realistic synthetic channels using sionna RT
    i=0
    batch_idx=0
    Dataset_list=[]
    pbar = tqdm(total=nb_batches, desc="Building Dataset", unit="batch")
    while batch_idx < nb_batches:
        channel=np.array((size_batches,nb_BS_antennas))

        # take nb_channels positions
        cur_loc=locs_grid[i*size_batches:(i+1)*size_batches,:]


        #Add UE 
        for idx in range(cur_loc.shape[0]):
            tx = Transmitter(name='tx'+f'_{idx}',
                    position=[cur_loc[idx][0],cur_loc[idx][1],cur_loc[idx][2]],
                        orientation=[0,0,0])
            scene.add(tx)

        paths=scene.compute_paths()
        paths.normalize_delays=False
        
        a, tau = paths.cir()
    
        frequencies=tf.cast(0,tf.float32) 

        
        h = sionna.channel.cir_to_ofdm_channel(frequencies, a, tau, normalize=False)
        h_temp = h.numpy().squeeze() #[nb_BS_antenna , nb_chan_train]

        channel=h_temp.T

        
        # test if any channel in the batch has 0 norm 
        if(0 in (np.linalg.norm(channel,2,axis=1))):
            print('channel will not be added to the current batch ')

        else:
            #Save data
            Dataset_list.append(channel) 
            pbar.update(1)  # manually update progress
            pbar.set_postfix(valid=batch_idx)
            batch_idx+=1
            
        #remove UE 
        for idx in range(cur_loc.shape[0]):
            scene.remove(f'tx_{idx}')         
        
        i+=1

    if Dataset_list:
        Dataset=np.vstack(Dataset_list)
        data={'Dataset':Dataset}
        np.savez( save_dir/ "Dataset.npz", **data)

    return Dataset


# generate DOA
def generate_DoA(save_dir,nb_DoA):
    DoA  =np.zeros((nb_DoA,3))
    DoA[:,1]= np.random.uniform(-np.pi,np.pi,nb_DoA)

    # save azimuth angles 
    doa={}
    doa['DoA']=DoA
    np.savez(save_dir/'DoA', **doa)


#Generate noisy channels with varying SNR
def generate_channel_realizations_varying_SNR(channels,noise_var,T):
    channels=torch.tensor(channels,dtype=torch.complex128)
    Nb_chan=channels.shape[0]
    Nb_BS_antenna=channels.shape[1]
    sigma_2 = torch.full([Nb_chan],noise_var) 
    h_noisy=torch.empty([Nb_chan,0])
    for t in range(T):
        n = (torch.sqrt(sigma_2).unsqueeze(-1))*(torch.randn(Nb_chan,Nb_BS_antenna)+1j*torch.randn(Nb_chan,Nb_BS_antenna)) # [N_u,A]
        h_noisy=torch.cat((h_noisy, channels+n), 1) # [N_u, AT] 
     
    return torch.tensor(h_noisy,dtype=torch.complex128),torch.tensor(sigma_2)        

def generate_channels_varying_SNR(path_init,Dataset,size_batches,nb_test_batches,nb_train_batches,noise_var,T):
    #normalize and permutate Dataset:
    norm_channels=np.linalg.norm(Dataset,axis=1)
    norm_max=np.max(norm_channels)
    norm_min=np.min(norm_channels)
    norm_factor=norm_max - norm_min
    Dataset=Dataset/norm_factor

    permut=np.random.permutation(Dataset.shape[0])
    Dataset=Dataset[permut]

    nb_channels_test= nb_test_batches*size_batches

    # TEST data
    test_data={}
    channel_test=Dataset[:nb_channels_test]
    #generate noisy channels,  the channel norm and sigma^2  
    h_noisy,noise_var_vect=generate_channel_realizations_varying_SNR(channel_test,noise_var,T)
    h=torch.tensor(channel_test,dtype=torch.complex128)
    test_data['h']          =h
    test_data['h_noisy']    =h_noisy
    test_data['sigma_2']    =noise_var_vect
    file_name = f"test_data.npz"  #Save data per batch
    save_dir=path_init / f'channels_var_snr/{noise_var:.0e}/T_{T}' 
    os.makedirs(save_dir,exist_ok=True)
    np.savez(save_dir/ file_name, **test_data) 
    
    # TRAIN data
    batch=0
    stop=False
    while batch<nb_train_batches and stop==False:
        #get channels
        train_data={}
        
        file_name=f"batch_{batch}.npz" 
        channel_train = Dataset[(batch*size_batches)+nb_channels_test:((batch+1)*size_batches)+nb_channels_test]
        #If number of channels in the batch is not enough 
        if channel_train.shape[0]!=size_batches:
            stop=True
        else:
            # generate noisy channels
            h_noisy,noise_var_vect=generate_channel_realizations_varying_SNR(channel_train,noise_var,T)
            h=torch.tensor(channel_train,dtype=torch.complex128)
            train_data['h']             =h#.repeat(1,T)
            train_data['h_noisy']       =h_noisy
            train_data['sigma_2']       =noise_var_vect
        
            #Save data
            np.savez(save_dir/ file_name, **train_data)
            batch=batch+1


#generate the measurement matrix M
def generate_M(path_init,nb_BS_antennas,nb_train_batches,L,T):
    M_dict={}
    M_test_dict={}

    save_dir= path_init/f'Measurement_matrix/L_{L}_T_{T}'
    os.makedirs(save_dir, exist_ok=True)

    # generate Measurement matrix for test 
    M_list=[]
    for t in range(T):
        phases = torch.rand(nb_BS_antennas, L) * (2 * torch.pi)
        m=torch.exp(1j* phases) # [LxA]
        M_list.append(m)

    M_test=torch.cat((M_list),1)
    if len(M_list)==1:
        M_tilde_test=M_test
    else: 
        M_tilde_test=torch.block_diag(*M_list)

    M_test=M_test.unsqueeze(0) #batched mx [*, TL, A]
    M_tilde_test=M_tilde_test.unsqueeze(0) #batched mx [*, TL, AT]

    M_test_dict['M_test']=M_test
    M_test_dict['M_tilde_test']=M_tilde_test

    #save data 
    np.savez(save_dir/ 'test.npz', **M_test_dict) 


    i=0

    while i<nb_train_batches:

        # generate Measurement matrix for test 
        M_list=[]
        for t in range(T):
            phases = torch.rand(nb_BS_antennas, L) * (2 * torch.pi)
            m=torch.exp(1j*phases) # [LxA]
            M_list.append(m)
        
        M_train=torch.cat((M_list),1)
        if len(M_list)==1:
            M_tilde_train=M_train
        else: 
            M_tilde_train=torch.block_diag(*M_list)

        M_train=M_train.unsqueeze(0)
        M_tilde_train=M_tilde_train.unsqueeze(0)

        M_dict['M_train']=M_train
        M_dict['M_tilde_train']=M_tilde_train

        #save data 
        np.savez(save_dir/ f"batch_{i}.npz", **M_dict) 

        i+=1