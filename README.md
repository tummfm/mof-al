# Active learning for partial charge prediction in MOFs

Code and data of the paper Active Learning Graph Neural Networks for Partial Charge Prediction of Metal-Organic Frameworks via Dropout Monte Carlo.

<p align="center">
<img src="https://drive.google.com/file/d/1llm-_JlPtqcUDInhcnW_82vH5O2SAswH/view?usp=sharing">
</p>


## Getting started
To run the pre-trained model obtained via active learning on structures 
from the ARC-MOF and Zeolite validation data, refer to ([run_trained_model.py](run_trained_model.py)).
 
To run the active learning, refer to [active_learning.py](active_learning.py). As a pre-requisite,
the data needs to be pre-processed using [data_preprocessing.py](data_preprocessing.py).

## Installation

All dependencies can be installed with the following two commands:
```
pip install -e .
pip install "jax[cuda]==0.3.14" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

```

## Contact

For questions, please open an issue on GitHub.

## Citation
Please cite our paper if you use our data, the trained models or this code
 in your own work:

```
@article{thaler_mof_2024,
  title = {Active Learning Graph Neural Networks for Partial Charge Prediction of Metal-Organic Frameworks via Dropout Monte Carlo},
  author = {Thaler, Stephan and Mayr, Felix and Thomas, Siby and Gagliardi, Alessio and Zavadlav, Julija},
  journal={npj Computational Materials},
  volume={},
  number={},
  pages={},
  doi={}
  year = {2024}
}
```