import numpy as np

class EarlyStopping:

    def __init__(self, patience=10, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None

        self.best_rmse = None
        self.best_recall = None
        self.best_map = None
        self.best_mae = None
        self.best_mape = None
        self.best_r_2 = None

        self.best_label = None
        self.best_pre = None

        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, current_val_loss,
                 test_rmse, test_recall, test_map, test_mae, test_mape, test_r_2,
                 test_inverse_trans_pre, test_inverse_trans_label):
        current_score = current_val_loss

        if self.best_score is None:
            self.best_score = current_score
            self.update_test_metrics(test_rmse, test_recall, test_map, test_mae, test_mape, test_r_2,
                                     test_inverse_trans_pre, test_inverse_trans_label)

        elif current_score > self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            print(f'EarlyStopping update val_loss: {self.best_score} --> {current_score}')
            self.best_score = current_score
            self.update_test_metrics(test_rmse, test_recall, test_map, test_mae, test_mape, test_r_2,
                                     test_inverse_trans_pre, test_inverse_trans_label)
            self.counter = 0

    def update_test_metrics(self, test_rmse, test_recall, test_map, test_mae, test_mape, test_r_2,
                            test_inverse_trans_pre, test_inverse_trans_label):
        self.best_rmse = test_rmse
        self.best_recall = test_recall
        self.best_map = test_map
        self.best_mae = test_mae
        self.best_mape = test_mape
        self.best_r_2 = test_r_2

        self.best_pre = test_inverse_trans_pre
        self.best_label = test_inverse_trans_label