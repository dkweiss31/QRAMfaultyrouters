# Faulty Towers: Recovering a functioning quantum random access memory in the presence of defective routers 

Code repository for [[arXiv]](https://arxiv.org/abs/2411.15612)

D. K. Weiss, Shifan Xu, Shruti Puri, Yongshan Ding, S. M. Girvin

## Install

To install the package, first clone the repository

```bash
git clone https://github.com/dkweiss31/QRAMfaultyrouters.git
```

Create a clean conda environment

```bash
conda create --name qram python=3.10
conda activate qram 
```

Move into the repository directory and install the package and its dependencies

```bash
cd QRAMfaultyrouters/
pip install -e .
```

## Run

One can then run the file `run_faulty_routers.py` which runs the `FlagQubitMinimization` algorithm on the last layer, the `IterativeRepair` algorithm on the whole tree and attempts `RelabelRepair` for all `3<=m<=n` where `n` is the tree depth. All of the simulation data is automatically written to file, for example to `00000_faulty_routers.h5py`. The simulation data can then be accessed by running

```python
from qram_repair import extract_info_from_h5

data_dict, param_dict = extract_info_from_h5("00000_faulty_routers.h5py")
```
All of the simulation data is in the `data_dict` dictionary, while all of the simulation data is in the `param_dict` dictionary.
