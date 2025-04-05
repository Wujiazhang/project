from data_loader import load_cifar10
from trainer import Trainer
from utils import  load_model
from visualization import (
    visualize_weights,
    visualize_w2_heatmap,
    plot_weight_distributions
)
from hyperparameter_tuning import hyperparameter_search

# ------------------------- 修改主程序 -------------------------#
if __name__ == '__main__':
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar10()
    best_params = hyperparameter_search(X_train, Y_train, X_val, Y_val)
    model = load_model()

    # 可视化第一层权重
    W1 = model.params['W1']
    visualize_weights(W1, "First Layer Weights", n_cols=16)
    # 可视化第二层权重
    W2 = model.params['W2']
    visualize_w2_heatmap(W2)

    # 可视化参数分布
    plot_weight_distributions(W1, W2)
    tester = Trainer(model, X_val, Y_val)  # 复用Trainer的评估方法
    test_acc = tester.check_accuracy(X_test, Y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # 额外验证加载模型的完整性
    val_acc = tester.check_accuracy(X_val, Y_val)
    print(f"Validation accuracy (for verification): {val_acc:.4f}")
