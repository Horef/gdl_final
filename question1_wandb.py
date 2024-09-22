import wandb
from question1_code import run_q1

run = wandb.init(
    project='gdl_final_project',
    config={
        'batch_size': [32, 64, 128],
        'epochs': 10,
        'lr': [0.001, 0.01, 0.1],
        'sample_points': [512, 1024, 2048]
    }
)

run_q1()