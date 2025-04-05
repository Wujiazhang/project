import numpy as np
import pickle
import os
def load_cifar10(data_dir='cifar-10-python/cifar-10-batches-py'):
    def load_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data'].astype('float32').reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).reshape(-1, 3072)
            Y = np.array(datadict['labels'])
            return X, Y

    # 加载训练集
    train_files = [os.path.join(data_dir, 'data_batch_%d' % i) for i in range(1, 6)]
    X_train, Y_train = [], []
    for f in train_files:
        X, Y = load_batch(f)
        X_train.append(X)
        Y_train.append(Y)
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)

    # 加载测试集
    X_test, Y_test = load_batch(os.path.join(data_dir, 'test_batch'))

    # 划分验证集（取训练集最后5000个样本）
    X_val = X_train[-5000:]
    Y_val = Y_train[-5000:]
    X_train = X_train[:-5000]
    Y_train = Y_train[:-5000]

    # 数据标准化
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, Y_train, X_val, Y_val, X_test, Y_test