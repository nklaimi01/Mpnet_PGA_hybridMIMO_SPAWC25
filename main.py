#%%
import subprocess
import numpy as np

# Generate log-spaced noise variances
noise_var_list = np.logspace(np.log10(6e-4), np.log10(2e-1), 6)
print("different noise variances used",[f"{noise_var:.0e}" for noise_var in noise_var_list])
#[6e-04, 2e-03, 6e-03, 2e-02, 6e-02, 2e-01]

L = 16
T = 1
# %% --------------------------- Run MpNet training ------------------------------------------
for noise_var in noise_var_list:
    result = subprocess.run(["python", "Run_MpNet.py", str(noise_var), str(L), str(T)], capture_output=True, text=True)
    print(result.stdout)  # Print captured output

    # subprocess.run(["python", "Run_MpNet.py", str(noise_var), str(L), str(T)])

# %%
