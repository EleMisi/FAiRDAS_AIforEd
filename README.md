# FAiRDAS_AIforEd

This repository provides the code for reproducing the results obtained in the paper 
*Ensuring Fairness Stability for Disentangling Social Inequality in Access to Education: the FAiRDAS General Framework* 
published at [IJCAI24](https://ijcai24.org/) (AI for Good track).

## Prerequisites :clipboard:

* Virtual environment with Python 3.7
* Packages:
  ```
  pip install -r requirements.txt
  ```
  
## How to Run  :arrow_forward:
The script `run.py` replicates the results presented in the paper:

  ```
  python run.py 
  ```

### Results 
Results are stored in `results` folder.\
In particular, the folder `results/records` contains:
* the configuration file with all the run information `config.json`
* the historical records of actions, metrics of interest, optimization and ranking (`.pkl` files)
* the mean and standard deviation of the metrics (`statistics.txt` and `statistics.json`) 

while the folder `results/images` contains images both in png and eps format.


### MLP Regressor
The MLP regressor has been trained on `utils/training_data.csv` with script  `utils/train_regressor.py`.
The resulting model weights are stored in `utils/regressor_checkpoint.pt`.

To re-train the regressor, run: 
  ```
  python utils/train_regressor.py
  ```

## Contacts :envelope:
Eleonora Misino: eleonora.misino2@unibo.it
