# PID-Controlled Anneal Langevin Dynamics Sampling Based on NCSNv2 -- 2-d Point Dataset Experiment


## Dependencies

Under the root directory of this project, create your python environment, and install the dependencies by running:
```bash
pip install -r requirements.txt
```


## Quick Start

Under the root directory of this project, run:
```bash
python run.py [options]
```
Command-line Arguments:

| Argument          | Type  | Default                | Description                     |
| ----------------- | ----- | ---------------------- | ------------------------------- |
| --seed            | int   | 42                     | Random seed for reproducibility |
| -P, --k_p         | float | 1.0                    | Propotional term coefficient    |
| -I, --k_i         | float | 0.0                    | Propotional term coefficient    |
| -D, --k_d         | float | 0.0                    | Propotional term coefficient    |
| --k_i_decay       | float | 1.0                    | Decay rate of k_i               |
| --result_dir      | str   | results                | Directory to save results       |
| --model_load_path | str   | scorenet_20_0.01_8.pth | Path to model weights           |

Examples:
```bash
python run.py
```
```bash
python run.py -I 0.1
```
```bash
python run.py -D 6.0
```
```bash
python run.py -I 0.1 -D 6.0
```
```bash
python run.py -I 0.1 -D 6.0 --k_i_decay 0.98
```
All results will be saved in `results` directory by default.

## Efficient Experiments

If you want to conduct multiple experiments by looping over different hyperparameters, you can customize the experiment loops by modifying `run2.py` and run it.

We also provide `main_light.py`, which is a light-weight version of `main.py` without excessive recording and visualization, suitable for massive experimentation. You can customize the experiment loops in `run2.py` to run efficient experiments.



For more details, please start from `main.py` or `main_light.py`. The structure and logic of this project are well documented in the comments.
