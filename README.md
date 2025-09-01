L'addestramento con Optuna viene avviato tramite lo script optuna_dcgan.py usando il comando  
python optuna_dcgan.py --dataset ~/dataset/Log3_real/BalancedTraining --validation ~/dataset/Log3/BalancedValidation --n_trials 70.

Per l'addestramento singolo si usa lo script run.py con il comando  
run.py: python run.py --gan dcgan --dataset ~/dataset/Log3_real/BalancedTraining -log

Le reti su cui stiamo lavorando ora sono TrainV2.py e dcgan_modelV2.py.
