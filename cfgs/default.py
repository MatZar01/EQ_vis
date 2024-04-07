model_info = {
    'DATA_PATH': 'DS/IDA-BD/i_B',
    'DEVICE': 'cuda',
    'EPOCHS': 60,
    'LR': 1e-4,
    'TRAIN_SIZE': 0.75,
    'FUSE_METHOD': 2,
    'DATA_SEED': 42,
    'BATCH_SIZE': 64,
    'OPT': 'Adam',
    'SCHEDULER': {'NAME': 'ROP', 'PATIENCE': 5, 'FACTOR': 0.8, 'STEP': 5, 'GAMMA': 0.5},
    'MODEL_NAME': 'Small_Net'
}