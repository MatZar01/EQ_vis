import sys
import importlib


def get_args() -> dict:
    """returns options for network training"""
    cfg_name = 'default'

    args = sys.argv

    if len(args) == 1:
        print(f'[INFO] No training arguments specified, using "{cfg_name}"')
    else:
        print(f'[INFO] using "{args[-1]}" training arguments')
        cfg_name = args[-1]

    try:
        cfg_module = importlib.import_module(f'cfgs.{cfg_name}')
        model_info = cfg_module.model_info
    except ModuleNotFoundError:
        model_info = None  # for consistency
        print(f'[ERROR] Config "{cfg_name}" not found in ./cfgs\n[INFO] Aborting')
        sys.exit()

    return model_info
