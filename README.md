# JEDI-linear


This repository contains the code the paper "JEDI-linear: Fast and Efficient Graph Neural Networks for Jet Tagging on FPGAs". The code can be run as follows:

1. Clone the repository, install the dependencies. Also install one of the backends for training (`jax`, `tensorflow`, or `pytorch`)
   ```bash
   pip install -r requirements.txt
   pip install <your_backend>
   ```
2. Download the dataset from https://zenodo.org/records/3602260
3. Extract the training and testing (the validation split downloaded), prepare them with
   ```bash
   python prepare_dataset.py -i /tmp/<train/validation>/ -o /tmp/<train/test>.h5 -j <n_processes>
   ```
   Place both `train.h5` and `test.h5` in the same directory, e.g., `/tmp/jet_data/`.
4. Modifying the configs to have the `datapath` point to the dataset directory, and change the output directory `save_path` if needed.
5. Run the training script:
   ```bash
   KERAS_BACKEND=<YOUR_BACKEND> python jet_classifier -c <CONFIG_FILE> -r train test verilog
   ```
   where `<YOUR_BACKEND>` can be `jax`, `tensorflow`, or `torch` depending on the backend you installed.
   The configs are located in `configs/gnn`.
   The `-n$number` part of the config file is the maximum number of particles to be used; `-3` means only `pt, eta, phi` are used, otherwise all 16 features are used; `uq1` means the network is uniformly quantized over the particle dimension and is permutation-invariant.
