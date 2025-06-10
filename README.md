# Model-based learning for joint channel estimation and hybrid MIMO precoding

Implementation of the methods proposed in the paper:

>📄 [Model-based learning for joint channel estimation and hybrid MIMO precoding](https://arxiv.org/abs/2505.04255)

> Nay Klaimi, Amira Bedoui, Clément Elvira, Philippe Mary, Luc Le Magoarou  
> SPAWC 2025
## Getting Started
### Install dependencies
<!-- requirements.txt -->

### Generate needed data from sionna library 
generate_data_sionna.py ==> generate_data_varying_snr.py ==> generate_M_DOA.py
<!-- write code that executes all of data generation -->

## 📚 Citation
Please consider citing the original paper if this code contributes to your work.
```bibtex
@misc{klaimi2025modelbasedlearningjointchannel,
      title={Model-based learning for joint channel estimationand hybrid MIMO precoding}, 
      author={Nay Klaimi and Amira Bedoui and Clément Elvira and Philippe Mary and Luc Le Magoarou},
      year={2025},
      eprint={2505.04255},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2505.04255}, 
}
```
<!-- Reminder: Update this section once the paper is published -->



<!--
mpnet
opt data load 
redundancy between run_mpnet and estimate_channels

uPGA
optmizie functions definition:
(DONE NEEDS TEST)"sum_loss" and "evaluate" functs
"plot_sum_rate" and "save_sum_rate" functions
avoid redundancy w/ plots.py 
opt data load
uPGA_true_channel

E2E 
optmizie functions definition:  "sum_loss" and "evaluate" functs
"plot_sum_rate" and "save_sum_rate" functions
in both E2E and E2E_naive files
avoid redundancy w/ plots.py 
opt data load

paper code
code thats plots figures for the paper
optimize paths-->
