model_info = {
    'DATA_PATH': 'DS/IDA-BD/i_B',
    'DEVICE': 'cuda',
    'EPOCHS': 30,
    'LR': 1e-4,
    'TRAIN_SIZE': 0.8,
    'FUSE_METHOD': 1,
    'DATA_SEED': 24,
    'BATCH_SIZE': 32,
    'NORMALIZE_INPUT': True,
    'OPT': 'Adam',
    'SCHEDULER': {'NAME': 'ROP', 'PATIENCE': 2, 'FACTOR': 0.8, 'STEP': 5, 'GAMMA': 0.5},
    'MODEL_NAME': 'Init_Net'
}