import numpy as np
np.random.seed(42)

class Trainer:
    def __init__(self, model, X_val, Y_val):
        self.model = model
        self.X_val = X_val
        self.Y_val = Y_val
        self.best_val_acc = 0.0
        self.best_params = {}
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def compute_loss(self, X, y, reg):
        probs = self.model.forward(X)
        corect_logprobs = -np.log(probs[range(X.shape[0]), y])
        data_loss = np.sum(corect_logprobs) / X.shape[0]
        reg_loss = 0.5 * reg * (np.sum(self.model.params['W1'] ** 2) + np.sum(self.model.params['W2'] ** 2))
        return data_loss + reg_loss

    def step(self, X_batch, y_batch, learning_rate, reg):
        self.model.forward(X_batch)
        grads = self.model.backward(X_batch, y_batch, reg)
        # 参数更新
        self.model.params['W1'] -= learning_rate * grads['W1']
        self.model.params['b1'] -= learning_rate * grads['b1']
        self.model.params['W2'] -= learning_rate * grads['W2']
        self.model.params['b2'] -= learning_rate * grads['b2']

    def check_accuracy(self, X, y):
        scores = self.model.forward(X)
        y_pred = np.argmax(scores, axis=1)
        return np.mean(y_pred == y)

    def train(self, X_train, Y_train, epochs=100, batch_size=200,
              learning_rate=1e-3, reg=1e-4, lr_decay=0.95, verbose=True):
        num_train = X_train.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        for epoch in range(epochs):
            # 学习率衰减
            if epoch % 10 == 0 and epoch != 0:
                learning_rate *= lr_decay

            # 随机打乱数据
            indices = np.random.permutation(num_train)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            for it in range(iterations_per_epoch):
                idx = it * batch_size
                X_batch = X_shuffled[idx:idx + batch_size]
                y_batch = Y_shuffled[idx:idx + batch_size]

                self.step(X_batch, y_batch, learning_rate, reg)

            # 验证集评估
            val_acc = self.check_accuracy(self.X_val, self.Y_val)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = {
                    'W1': self.model.params['W1'].copy(),
                    'b1': self.model.params['b1'].copy(),
                    'W2': self.model.params['W2'].copy(),
                    'b2': self.model.params['b2'].copy()
                }
            train_loss = self.compute_loss(X_train, Y_train, reg)
            val_loss = self.compute_loss(self.X_val, self.Y_val, reg)
            train_acc = self.check_accuracy(X_train, Y_train)
            val_acc = self.check_accuracy(self.X_val, self.Y_val)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            if verbose and epoch % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}: train loss {train_loss:.4f},  val loss {val_loss:.4f},"
                      f"train acc {train_acc:.4f}, val acc {val_acc:.4f}", flush=True)

        # 恢复最佳参数
        self.model.params = self.best_params

        return self.history