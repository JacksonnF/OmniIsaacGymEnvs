# README.md 
#### *Kyle Mackenzie* 

## Common commands

Training, no testing:

`PYTHON_PATH scripts/rlgames_train.py task=Broomy`

Testing, using a checkpoint:

`PYTHON_PATH scripts/rlgames_train.py task=Broomy test=True checkpoint=runs/Broomy/nn/Broomy.pth num_envs=64
`

Exporting:

`PYTHON_PATH export.py
`
