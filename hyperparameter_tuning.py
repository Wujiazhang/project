from model import ThreeLayerNet
from trainer import Trainer
from utils import save_model
from visualization import plot_training_curves
def hyperparameter_search(X_train, Y_train, X_val, Y_val):
    best_acc = 0.0
    best_params = {}

    # 搜索空间
    hidden_sizes = [256, 512]
    learning_rates = [1e-3, 5e-4]
    reg_strengths = [1e-4, 5e-4]

    for hs in hidden_sizes:
        for lr in learning_rates:
            for reg in reg_strengths:
                model = ThreeLayerNet(3072, hs, 10, activation='relu')
                trainer = Trainer(model, X_val, Y_val)
                history = trainer.train(X_train, Y_train, epochs=100,
                              batch_size=64, learning_rate=lr, reg=reg)
                val_acc = trainer.check_accuracy(X_val, Y_val)
                plot_training_curves(history, hs, lr, reg)
                print(f"hs: {hs}, lr: {lr}, reg: {reg} => val_acc: {val_acc:.4f}")
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = {'hidden_size': hs, 'lr': lr, 'reg': reg}
                    save_model(model, 'trained_model.npz')

    print("\nBest validation accuracy: %.4f" % best_acc)
    print("Best parameters:", best_params)
    return best_params