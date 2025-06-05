import numpy as np
import tensorflow as tf
import torch# Import Sionna RT components
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from sionna.channel import cir_to_ofdm_channel
from pathlib import Path
import matplotlib.pyplot as plt
import os
path_init = Path.cwd()/'.saved_data'
# save_dir=path_init/'Results'
# os.makedirs(save_dir,exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_scene(BS_position,f0):
    '''
    generate scene that contain  BS position 
    
    input: tx_pos => corresponds to the user's positions
           rx_pos => The BS position 
    
    output: scene
            nominal antenna positions
            real antenna positions
    
    '''
    
    scene = load_scene(sionna.rt.scene.etoile)
    nb_BS_antenna=64
    #nb_BS_antenna=256

    # change frequency
    scene.frequency=f0
    lambda_ = 0.010706874

    
    # Configuration of transmitters
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")
    
    # Configuration of recievers 
    # the antenna by default is located in the y-z plane 
    # This config creates antennas that are aligned over y axis 
    scene.rx_array = PlanarArray(num_rows=1, num_cols=nb_BS_antenna,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")
    
    
    #generate real and nominal positions
    nominal_ant_positions = tf.Variable(scene.rx_array.positions)
    scene.rx_array.positions=tf.Variable(scene.rx_array.positions)
    #print(scene.rx_array.positions)
    scene.rx_array.positions[:,1].assign(scene.rx_array.positions[:,1]+ 0.1*lambda_*np.random.randn(nb_BS_antenna))
    real_ant_positions = scene.rx_array.positions
    #print(scene.rx_array.positions)
    
    
    # Add BS to the scene
    rx = Receiver("rx", position=BS_position, orientation=[0,0,0])
    scene.add(rx)
    

        
    return scene,nominal_ant_positions,real_ant_positions 


def preprocess(data_path,batch_size):
    #Preprocessing:Dataset normalization 
    
    batch_idx=1
    path_init = Path.cwd()/'.saved_data'

    file_name=f"batch_0.npz" 
    data = np.load(path_init/data_path/file_name)
    channel_0= data['channel_train']

    Dataset=channel_0


    while(batch_idx<batch_size):
        
        #get channels       
        file_name=f"batch_{batch_idx}.npz" 
        data = np.load(path_init/data_path/file_name)
        channel_train= data['channel_train']

        Dataset=np.vstack((Dataset,channel_train))      
        batch_idx+=1
        
    norm_channels=np.linalg.norm(Dataset,axis=1)
    norm_max=np.max(norm_channels)
    norm_min=np.min(norm_channels)
    norm_factor=norm_max - norm_min
    
    return norm_factor

def plot_learning_curve(name, train_losses,
                        valid_losses,
                        num_of_iter_pga_unf,
                        epochs,
                        batch_size):
        y_t = [r.detach().numpy() for r in train_losses]
        x_t = np.array(list(range(len(train_losses))))
        y_v = [r.detach().numpy() for r in valid_losses]
        x_v = np.array(list(range(len(valid_losses))))
        plt.figure()
        plt.plot(x_t, y_t, 'o-', label='Train')
        plt.plot(x_v, y_v, '*-', label='Valid')
        plt.grid()
        plt.title(f'Loss Curve, Num Epochs = {epochs}, Batch Size = {batch_size} \n Num of Iterations of PGA = {num_of_iter_pga_unf}')
        plt.xlabel('Epoch')
        plt.legend(loc='best')               
        # plt.savefig(save_dir/name,dpi=500)
        plt.xlim(left=0)
        plt.show()

# PGA functions :

def sum_loss(wa, wd, h, U, batch_size, noise_var_DL):
    a1 = torch.transpose(wa, 1, 2).conj() @ torch.transpose(h, 1, 2).conj()
    a2 = torch.transpose(wd, 1, 2).conj() @ a1
    a3 = h @ wa @ wd @ a2
    g = torch.eye(U,device=device).reshape((1, U, U)) + a3 / (U * noise_var_DL)  # g = Ik + H*Wa*Wd*Wd^(H)*Wa^(H)*H^(H)
    s = torch.log(g.det())  # s = log(det(g))

  
    loss = sum(torch.abs(s)) / batch_size
    return -loss

# Define evaluation function 
def evaluate_sum_rate(h, WA, WD, U, noise_var_DL, batch_size, num_iter):
    sum_rate = torch.zeros(num_iter+1)
    for i in range(num_iter+1):
        sum_rate[i] = sum_loss(WA[i], WD[i], h, U, batch_size, noise_var_DL)
       
        
    return -sum_rate
 
# ploting the results
def plot_sum_rate(sum_rate,pga_type,channel_type,A,U,L,T,sum_rate_0=None):
    y = sum_rate.cpu().detach().numpy()
    x = np.arange(len(y)) 

    plt.figure()
    plt.plot(x, y, '+-',label='after training')
    if sum_rate_0 is not None:
        y0=sum_rate_0.cpu().detach().numpy()
        plt.plot(x, y0, '+--',color='grey',label='before training')
        
    plt.title(f'Sum rate, {pga_type}, {channel_type} A={A} U={U} L={L} T={T}')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Achievable Rate')
    plt.grid()
    plt.legend()
    plt.xlim(left=0)
    plt.show()


def save_sum_rate(file_name,sum_rate_,noise_var_DL,L,T):
    
    # save sum rate
    sum_rate={}
    sum_rate[f'{file_name}']=sum_rate_.detach().numpy()
    save_dir=path_init / 'sumRate'/f'{noise_var_DL:.0e}/L_{L}_T_{T}'
    os.makedirs(save_dir,exist_ok=True)
    np.savez(save_dir/f'{file_name}', **sum_rate)