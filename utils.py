import numpy as np
from model import ThreeLayerNet
import os
def save_model(model, filename='trained_model.npz'):
    """保存模型权重到文件"""
    np.savez(
        filename,
        W1=model.params['W1'],
        b1=model.params['b1'],
        W2=model.params['W2'],
        b2=model.params['b2'],
        input_size=model.input_size,
        hidden_size=model.hidden_size,
        output_size=model.output_size,
        activation=model.activation
    )
    print(f"Model saved to {filename}")


def load_model(filename='trained_model.npz'):
    """从文件加载模型权重"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No model file {filename} found")

    data = np.load(filename, allow_pickle=True)
    model = ThreeLayerNet(
        input_size=int(data['input_size']),
        hidden_size=int(data['hidden_size']),
        output_size=int(data['output_size']),
        activation=str(data['activation'])
    )

    model.params = {
        'W1': data['W1'],
        'b1': data['b1'],
        'W2': data['W2'],
        'b2': data['b2']
    }
    print(f"Model loaded from {filename}")
    return model
