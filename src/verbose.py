def verbose(epoch: int, train_res: list, test_res: list, freq: int = 25):
    """Outputs to terminal"""
    epoch += 1
    if epoch % freq == 0:
        print(f'[INFO] Epoch: {epoch}, Train loss: {train_res[0]}, Train acc: {train_res[-1]}')
        print(f'[INFO] Test loss: {test_res[0]}, Test acc: {test_res[-1]}')

