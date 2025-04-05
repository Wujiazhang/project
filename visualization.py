import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history, hs, lr, reg):

    plt.figure(figsize=(12, 4))
    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'training_curves_{hs}_{lr}_{reg}.png')
    plt.show()


def visualize_weights(weights, title, n_cols=16, channel_order=(0, 1, 2)):
    """
    可视化第一层权重
    :param weights: 权重矩阵 shape (input_dim, hidden_dim)
    :param title: 图像标题
    :param n_cols: 每行显示的数量
    """
    # 归一化权重到[0,1]
    weights = weights - weights.min()
    weights /= weights.max()

    n_filters = weights.shape[1]
    n_rows = int(np.ceil(n_filters / n_cols))

    plt.figure(figsize=(n_cols, n_rows * 1.1))
    for i in range(n_filters):
        plt.subplot(n_rows, n_cols, i + 1)
        # 调整通道顺序为RGB并reshape为32x32x3
        w_img = weights[:, i].reshape(32, 32, 3)[:, :, channel_order]
        plt.imshow(w_img)
        plt.axis('off')
    plt.suptitle(title)
    plt.savefig('first_layer_weights.png')
    plt.show()


def visualize_w2_heatmap(weights, class_names=None):
    """
    可视化第二层权重（隐藏层到输出层的权重矩阵）

    参数：
    W2 : numpy.ndarray
        权重矩阵，形状为 (hidden_size, num_classes)
    class_names : list, optional
        类别名称列表，默认为CIFAR-10的官方类别
    """
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(12, 8))

    # 归一化权重到[-1, 1]范围
    max_val = np.max(np.abs(weights))
    normalized_weights = weights / max_val

    # 创建热力图（转置矩阵使类别在x轴）
    heatmap = plt.imshow(normalized_weights.T,  # 转置矩阵
                         cmap='coolwarm',
                         aspect='auto',
                         vmin=-1,
                         vmax=1,
                         interpolation='nearest')

    # 添加标签和标题
    plt.xlabel("Hidden Units", fontsize=12)
    plt.ylabel("Classes", fontsize=12)
    plt.title("W2 Weight Matrix Heatmap", fontsize=14)

    # 设置坐标轴
    plt.xticks([])  # 隐藏x轴刻度（隐藏单元数量通常太多）
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=10)

    # 添加颜色条
    cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Weight Value", rotation=270, labelpad=15)

    # 添加网格线
    for i in range(len(class_names)):
        plt.axhline(i + 0.5, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('w2_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_weight_distributions(W1, W2):

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(W1.flatten(), bins=200)
    plt.title("W1 Weight Distribution")

    plt.subplot(1, 2, 2)
    plt.hist(W2.flatten(), bins=200)
    plt.title("W2 Weight Distribution")
    plt.savefig('weight_distributions.png')
    plt.show()
