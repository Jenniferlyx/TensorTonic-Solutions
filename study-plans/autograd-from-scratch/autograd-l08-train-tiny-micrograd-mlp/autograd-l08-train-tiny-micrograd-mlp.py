import torch

def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    """
    Returns final predictions, final loss, trained weights, trained biases, and pre-update loss history.
    """
    weights = [w.clone() for w in weights]
    biases = [b.clone() for b in biases]
    def forward():
        activations = [inputs]
        for w, b in zip(weights, biases):
            h = torch.tanh(activations[-1] @ w.T + b)
            activations.append(h)
        return activations
    loss_history = []
    for _ in range(steps):
        # 1. Forward pass
        activations = forward()
        predictions = activations[-1].squeeze(-1)
        # 2. Loss
        loss = ((predictions - targets) ** 2).sum()
        loss_history.append(loss)
        # 3. Backward pass: start with dL/d(predictions)
        dA = (2 * (predictions - targets)).unsqueeze(-1)
        weight_grads = [None] * len(weights)
        bias_grads = [None] * len(biases)
        for layer in reversed(range(len(weights))):
            A = activations[layer + 1] # this layer's output
            H = activations[layer] # this layer's input
            dZ = dA * (1 - A ** 2)

            dW = dZ.T @ H
            db = dZ.sum(dim=0)
            weight_grads[layer] = dW
            bias_grads[layer] = db
            dA = dZ @ weights[layer]
        # 4. Update all parameters
        for layer in range(len(weights)):
            weights[layer] = weights[layer] - learning_rate * weight_grads[layer]
            biases[layer] = biases[layer] - learning_rate * bias_grads[layer]
    predictions = forward()[-1].squeeze(-1)
    final_loss = ((predictions - targets) ** 2).sum()
    return predictions, final_loss, weights, biases, loss_history
        
        
        
