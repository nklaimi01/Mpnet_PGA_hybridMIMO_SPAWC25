#%%################################################################################################
###########---------------------NMSE vs nb of seen channels ------------------##################
################################################################################################

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
# import Mpnet_training 

path_init=Path.cwd().parent/'.saved_data'
save_dir=path_init/'paper_mpnet_models'

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import Mpnet_training

# specify parameters
noise_var= 2e-3 #SNR_ul=15 dB
# noise_var= 2e-2 #SNR_ul=5 dB
# RF chains 
L = 16
T = 3
print(f"MpNet training with noise_var={noise_var:.0e}, L={L}, T={T}")

#load mpnet trained models
model=torch.load(save_dir/f'model_{noise_var:.0e}_L_{L}_T_{T}.pth')
model_sup=torch.load(save_dir/f'model_sup_{noise_var:.0e}_L_{L}_T_{T}.pth')
# plot figure
plt.rcParams['text.usetex'] = True
plt.figure()

# Ensure nominal MP values are set correctly
NMSE_mp_nominal = np.ones_like(model.NMSE_mp_nominal) * model.NMSE_mpnet_test[0]  # Same for both models

# Define colors
mpnet_sup_color = 'tab:blue'        # mpNet Unsup & Sup
mpnet_unsup_color = 'tab:orange'  # mpNet Constrained Unsup & Sup
nominal_color = 'tab:purple'    # MP Nominal
lmmse_color = 'tab:red'         # LMMSE
mp_real_color = 'tab:green'     # MP Real

# lower bounds
x_reduced = np.unique(np.concatenate([np.arange(0, len(model.NMSE_mpnet_test), step=5), [len(model.NMSE_mpnet_test) - 1]])) 
plt.plot(x_reduced,model.NMSE_lmmse[x_reduced], 'x-', color=lmmse_color, linewidth=1.5, label='LMMSE')
plt.plot(NMSE_mp_nominal, ':', color=nominal_color, linewidth=1.5, label='MP (Nominal dictionary)')  

# Plot results
plt.plot(model.NMSE_mpnet_test, '.-', color=mpnet_unsup_color, linewidth=1, label='mpNet Unsupervised')  
plt.plot(model_sup.NMSE_mpnet_test, '--', color=mpnet_sup_color, linewidth=1, label='mpNet Supervised')  

plt.plot(model.NMSE_mpnet_test_c, '+-', color=mpnet_unsup_color, linewidth=1, label='mpNet Constrained Unsupervised')
plt.plot(model_sup.NMSE_mpnet_test_c, '-', color=mpnet_sup_color, linewidth=1, label='mpNet Constrained Supervised')


# upper bound
plt.plot(model.NMSE_mp_real, '-.', color=mp_real_color, linewidth=1.5, label='MP (Real dictionary)')

plt.grid()
plt.xlabel(r'Number of seen channels $(10^3)$',fontsize=16)
plt.ylabel('NMSE',fontsize=14)

# plt.title('NMSE Evolution vs number of seen channels')
plt.xlim(left=0)
# plt.xlim(right=3000)
plt.ylim(0, 1)
plt.tick_params(axis='both', labelsize=16) # Increase both x and y tick label size
plt.yticks([tick for tick in plt.yticks()[0] if tick != 0]) #remove 0 from y-axis tick labels

# plt.tick_params(labelleft=False) # remove ylabels
plt.legend(loc='upper right') # show legend

plt.savefig(f"nmse_{noise_var:.0e}_L_{L}_T_{T}.pdf", format="pdf", bbox_inches="tight")

plt.show()
# %% 
######################################## subplot #####################################################
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# === Préparation ===
path_init = Path.cwd().parent / '.saved_data'
save_dir = path_init / 'paper_mpnet_models'
sys.path.append(str(Path(__file__).resolve().parent.parent))
import Mpnet_training

# === Paramètres des 3 cas ===
configs = [
    {'T': 3, 'noise_var': 2e-3},
    {'T': 1, 'noise_var': 2e-3},
    {'T': 1, 'noise_var': 2e-2},
]

titles = [
    r"(a) $T=3,\ L=16,\ \mathrm{SNR}_{\mathrm{av,UL}}=15\ \mathrm{dB}$",
    r"(b) $T=1,\ L=16,\ \mathrm{SNR}_{\mathrm{av,UL}}=15\ \mathrm{dB}$",
    r"(c) $T=1,\ L=16,\ \mathrm{SNR}_{\mathrm{av,UL}}=5\ \mathrm{dB}$",
]

# === Styles ===
colors = {
    'mpnet_sup': 'tab:blue',
    'mpnet_unsup': 'tab:orange',
    'nominal': 'tab:purple',
    'lmmse': 'tab:red',
    'mp_real': 'tab:green'
}

# === Subplots ===
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, (cfg, ax) in enumerate(zip(configs, axs)):
    noise_var = cfg['noise_var']
    T = cfg['T']
    L = 16

    # Load models
    model = torch.load(save_dir / f'model_{noise_var:.0e}_L_{L}_T_{T}.pth')
    model_sup = torch.load(save_dir / f'model_sup_{noise_var:.0e}_L_{L}_T_{T}.pth')
    NMSE_mp_nominal = np.ones_like(model.NMSE_mp_nominal) * model.NMSE_mpnet_test[0]

    # Plot
    x_reduced = np.unique(np.concatenate([
        np.arange(0, len(model.NMSE_mpnet_test), step=5),
        [len(model.NMSE_mpnet_test) - 1]
    ]))

    ax.plot(x_reduced, model.NMSE_lmmse[x_reduced], 'x-', color=colors['lmmse'], linewidth=1.5, label='LMMSE')
    ax.plot(NMSE_mp_nominal, ':', color=colors['nominal'], linewidth=1.5, label='MP (Nominal dict.)')
    ax.plot(model.NMSE_mpnet_test, '.-', color=colors['mpnet_unsup'], linewidth=1, label='mpNet Unsupervised')
    ax.plot(model_sup.NMSE_mpnet_test, '--', color=colors['mpnet_sup'], linewidth=1, label='mpNet Supervised')
    ax.plot(model.NMSE_mpnet_test_c, '+-', color=colors['mpnet_unsup'], linewidth=1, label='mpNet Constrained Unsup')
    ax.plot(model_sup.NMSE_mpnet_test_c, '-', color=colors['mpnet_sup'], linewidth=1, label='mpNet Constrained Sup')
    ax.plot(model.NMSE_mp_real, '-.', color=colors['mp_real'], linewidth=1.5, label='MP (Real dict.)')

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_title(titles[i], fontsize=16)
    ax.set_yticks([t for t in ax.get_yticks() if t != 0])
    ax.set_xlabel(r'Number of seen channels $(10^3)$', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    if i == 0:
        ax.set_ylabel('NMSE', fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
    else:
        ax.tick_params(labelleft=False)  # remove y tick labels

# === Légende ===
handles, labels = axs[0].get_legend_handles_labels()

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("nmse_comparison_subplot.pdf", format="pdf", bbox_inches="tight")
plt.show()


# %%
