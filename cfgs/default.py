model_info = {
    'DATA_PATH': 'DS/IDA-BD/i_B',
    'ONEHOT_DATA': True,
    'DEVICE': 'cuda',
    'EPOCHS': 60,
    'LR': 1e-4,
    'TRAIN_SIZE': 0.8,
    'FUSE_METHOD': 2,
    'DATA_SEED': 24,
    'BATCH_SIZE': 32,
    'NORMALIZE_INPUT': True,
    'OPT': 'Adam',
    'SCHEDULER': {'NAME': 'ROP', 'PATIENCE': 2, 'FACTOR': 0.7, 'STEP': 5, 'GAMMA': 0.5},
    'MODEL_NAME': 'ResNet50_Fuse_Net',
    'LOG': True,

    'CLASS_W': [0.7298184429172223, 2.1619568520206625, 6.197735191637631, 169.4047619047619]
}
