import wandb

wandb_username = 'gdl_final_team'
wandb_project = 'gdl_final_project'

command = ['${env}', '${interpreter}', 'q2_code.py', '${args}']

sweep_config={
    'method': 'grid',
    'metric': {
        'name': 'Val/Max_Accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'value': 100
        },
        'lr': {
            'values': [0.0005, 0.0007, 0.001, 0.005, 0.01]
        },
        'n_layer': {
            'values': list(range(1, 11))
        },
        'agg_hidden': {
            'values': [8, 16, 32, 64, 128, 256]
        },
        'fc_hidden': {
            'values': [8, 16, 32, 64, 128, 256]
        },
        'wandb': {
            'value': 1
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