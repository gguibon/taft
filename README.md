# TAFT: Task-Adaptive Fine-Tuning

This directory contains the anonymous code for our paper "An Adaptive Layer to Leverage both Domain and Task Specific Information from Scarce Data" published at AAAI 2023.
This code allows you to run our model (TAFT), its variants (TAFT noAdapt), and baselines (DAPT, TAPT) on GPUs. Please refer to the paper and Table 2 for more information.

Jump to 
```console
bash runs.sh
```
to run the experiments or open this file to see the commands for each experiment

## Citation

If you find this work useful, please cite the following paper:
```latex
@inproceedings{guibon2023adapt,
  title={An Adaptive Layer to Leverage both Domain and Task Specific Information from Scarce Data},
  author={Guibon, Ga{\"e}l and Labeau, Matthieu and Lefeuvre, Luce and Clavel, Chlo{\'e}},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

## Requirements

To run this code you need some python imports. Please refer to the `requirements.txt` file to see the full list of our Python environnement used for the experiments, along with their version.
The main imports are of course PyTorch and affiliates, PyTorch lightning, Scikit-learn and Hugging Face's Tranformers.

Running the scripts and the training will require disk storage to save the models. It will automatically create some directories with new files (models, tensorboard logs, plots, and reports).

GPU environnement inforamtion are as follows (output from `nvidia-smi` command):
```console
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100S-PCI...  On   | 00000000:25:00.0 Off |                    0 |
| N/A   27C    P0    24W / 250W |      4MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

## Dummy Dataset
Given we cannot legally share our confidential customer service dataset, we use here a dummy dataset that mimmics the data preparation process and the exact original data structure.

## Folder organization

The code structure leverages PyTorch Lightning to obtain pseudo scripts dedicated to specific experiments. We will improve it to remove unecessary repetitions later on (a bit of factorization is needed).
Even though, the standard organization is a `CustomDataModule` class to deal with dataset related info, and a `PLBert` class to deal with model building.

The folder is organized as follows:

```shell
.
├── dapt_mlm.py ==> DAPT STEP 1
├── dapt_satisfaction_ft.py ==> DAPT STEP 2 ON SATISFACTION
├── dapt_status_ft.py == DAPT STEP 2 ON EITHER PC OR STATUS TASKS
├── data ==> DIRECTORY FOR INPUT DATA
│   ├── dummy_dataset.json ==> DUMMY DATASET
│   └── generate_fake_dataset.py ==> DUMMY DATASET GENERATOR AND PREPROCESSING
├── dataset_utils ==> UTILITARIES FOR DATASET HANDLING
│   ├── categories_sampler.py ==> SAMPLER BY CATEGORIES (FOR BALANCED DISTRIBUTION)
│   ├── custom_dataset.py 
│   ├── episodic_sampler.py ==> EPISODIC SAMPLER AS TACKLED IN THE PAPER
│   └── imbalanced_sampler.py ==> IMBALANCED SAMPLER USED TO REPORT THE RESULTS IN THE PAPER
├── README.md
├── requirements.txt ==> PYTHON ENVIRONNEMENT
├── runs.sh ==> MAIN START FILE TO RUN THE EXPERIMENTS
├── speaker_role_pretraining.py ==> TAFT OR TAPT STEP 1
├── taft_target_finetuning.py ==> TAFT STEP 2 WITH MULTITASK FINETUNING OF THE ADAPTIVE LAYER
└── utils.py

3 directories, 18 files
```


## How to run the experiments

Commands lines dedicated to each experiments with sections are provided in the `runs.sh` Bash script. You only need to comment the commands you do not want to try.
Each command is linked with a comment explaining the dedicated step from Figure 1 in the paper.

```console
bash runs.sh
```



