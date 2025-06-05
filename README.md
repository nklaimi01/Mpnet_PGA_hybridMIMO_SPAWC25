## joint_mpNet_uPGA




# Generate needed data from sionna library 
generate_data_sionna.py ==> generate_data_varying_snr.py ==> generate_M_DOA.py
write code that ecutes all of data generation (?)

# mpnet
opt data load 
redundancy between run_mpnet and estimate_channels

# uPGA
optmizie functions definition:
(DONE NEEDS TEST)"sum_loss" and "evaluate" functs
"plot_sum_rate" and "save_sum_rate" functions
avoid redundancy w/ plots.py 
opt data load
uPGA_true_channel

# E2E 
optmizie functions definition:  "sum_loss" and "evaluate" functs
"plot_sum_rate" and "save_sum_rate" functions
in both E2E and E2E_naive files
avoid redundancy w/ plots.py 
opt data load

# paper code
code thats plots figures for the paper

optimize paths