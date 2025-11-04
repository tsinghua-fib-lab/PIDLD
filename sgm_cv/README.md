# PID-Controlled Annealed Langevin Dynamics Sampling Based on NCSNv2 -- Computer Vision Task


## Dependencies

Under the root directory of this project, create your python environment, and install the dependencies by running:
```bash
pip install -r requirements.txt
```


## Quick Start

To run the project, you should download `model_weights` from <a href="https://drive.google.com/drive/folders/1H1vVFOtLnaZw1LmNrH56gKrwvhlh0MB2?usp=drive_link">this link</a> and put it in the root directory of the project. These checkpoints were originally downloaded from <a href="https://drive.google.com/drive/folders/1217uhIvLg9ZrYNKOR3XTRFSurt4miQrd?usp=sharing">Yang Song's sharing</a>, which is found at <a href="https://github.com/ermongroup/ncsnv2">NCSNv2 repository</a>.

The Project structure is:
```sh
root
├── configs
│   ├── celeba.yml
│   └── cifar10.yml
├── evaluation
│   ├── inception.py
│   └── metrics.py
├── exp # directory created to save experiment data
│   ├── exp1
│   ├── exp2
│   ├── exp3
│   └── ...
├── model_weights # neural network weights and inception statistics
│   ├── celeba
|   |   ├── best_checkpoint_with_denoising.pth
|   |   └── celeba_test_fid_stats.npz
│   ├── cifar10
|   |   ├── best_checkpoint_with_denoising.pth
|   |   └── fid_stats_cifar10_train.npz
│   └── hub
|       └── checkpoints
|           └── pt_inception-2015-12-05-6726825d.pth
├── models
│   ├── ema.py
│   ├── layers.py
│   ├── refinenet.py
│   └── normalization.py
├── utils
│   ├── __init__.py
│   ├── format.py
│   ├── hooks.py
│   └── log.py
├── dynamics.py
├── main.py
└── README.md
```

Command line arguments for `main.py` are as follows:

| Argument              |          Type | Default             | Description                                                                                                   |
| --------------------- | ------------: | ------------------- | ------------------------------------------------------------------------------------------------------------- |
| `--config`            |           str | (required)          | Path to the configuration file (located in the `configs` directory). Example: `--config cifar10.yml`          |
| `--seed`              |           int | 1234                | Random seed for reproducibility. Example: `--seed 4321`                                                       |
| `--exp`               |           str | exp                 | Path for saving experiment-related data. Example: `--exp ./experiment_data`                                   |
| `--comment`           |           str | `''` (empty string) | Comment string for the experiment to record additional info. Example: `--comment 'This is a test experiment'` |
| `--exp_name`          |           str | default             | Experiment name. Example: `--exp_name my_experiment`                                                          |
| `--exp_dir_suffix`    |   str \| None | `None`              | Suffix added to the experiment directory to distinguish experiments. Example: `--exp_dir_suffix v2`           |
| `-P`/`--k_p`          | float \| None | 1.0                 | Coefficient for Proportional Gain. Example: `-P 0.1`                                                          |
| `-I`/`--k_i`          | float \| None | 0.0                 | Coefficient for Integral Gain. Example: `-I 0.01`                                                             |
| `-D`/`--k_d`          | float \| None | 0.0                 | Coefficient for Differential Gain. Example: `-D 0.001`                                                        |
| `--k_i_decay`         | float \| None | 1.0                 | Decay rate for Integral Gain. Example: `--k_i_decay 0.99`                                                     |
| `--k_d_decay`         | float \| None | 1.0                 | Decay rate for Differential Gain. Example: `--k_d_decay 0.98`                                                 |
| `-T`/`--n_steps_each` |   int \| None | `None`              | Number of sampling steps per noise level. Example: `-T 100`                                                   |
| `-L`/`--num_classes`  |   int \| None | `None`              | Number of noise levels (classes). Example: `-L 10`                                                            |


For example, to sample CIFAR10 images with $k_p=2.0$, $k_i=0.5$, $k_d=4.5$, $100$ noise levels, and $1$ step per noise level, run
```sh
python main.py --config cifar10.yml --exp_name k_p=2.0_k_i=0.5_k_d=4.5_100x1_steps -P 2.0 -I 0.5 -D 4.5 -L 100 -T 1
```
