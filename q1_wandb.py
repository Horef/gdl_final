import wandb

wandb_username = 'gdl_final_team'
wandb_project = 'gdl_final_project'

command = ['${env}', '${interpreter}', 'q1_code.py', '${args}']

sweep_config={
    'method': 'grid',
    'metric': {
        'name': 'Test/Accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'batch_size': {
            'values': [64, 128, 256]
        },
        'epochs': {
            'value': 10
        },
        'lr': {
            'values': [0.001, 0.01, 0.1]
        },
        'sample_points': {
            'values': [512, 1024, 2048]
        }
    },
    'command': command
}

sweep_id = wandb.sweep(sweep_config, project=wandb_project)

print("Run these lines to run your agent in a screen:")
parallel_num = 8

if parallel_num > 10:
    print('Are you sure you want to run more than 10 agents in parallel? It would result in a CPU bottleneck.')
for i in range(parallel_num):
    print(f"screen -dmS \"sweep_agent_{i}\" wandb agent {wandb_username}/{wandb_project}/{sweep_id}")