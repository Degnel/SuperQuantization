import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from quantized_net import QuantizedNet
from full_net import FullPrecisionNet


def raw_test():
    input_dim = 20
    hidden_dim = 20
    batch_size = 64000
    depths_full = range(1, 7)
    depths_quant = [d * 8 for d in depths_full]

    data = torch.randn(batch_size, input_dim)

    mse_results_full_to_quant = []
    mse_results_quant_to_full = []

    for depth_full, depth_quant in zip(depths_full, depths_quant):
        print("depth_full: ", depth_full, " depth_quant: ", depth_quant)

        full_model = FullPrecisionNet(depth_full, input_dim, hidden_dim)
        quant_model = QuantizedNet(depth_quant, input_dim, hidden_dim)

        mse_full_to_quant = train_model(full_model, quant_model, data, True)
        mse_results_full_to_quant.append(mse_full_to_quant)
        print("MSE Full to Quantized: ", mse_full_to_quant)

        full_model = FullPrecisionNet(depth_full, input_dim, hidden_dim)
        quant_model = QuantizedNet(depth_quant, input_dim, hidden_dim)

        mse_quant_to_full = train_model(quant_model, full_model, data, False)
        mse_results_quant_to_full.append(mse_quant_to_full)
        print("MSE Quantized to Full: ", mse_quant_to_full)

    plt.figure(figsize=(10, 6))
    plt.plot(depths_full, mse_results_full_to_quant, label="Quantized imitates Full")
    plt.plot(depths_full, mse_results_quant_to_full, label="Full imitates Quantized")
    plt.xlabel("Depth of Full Precision Network")
    plt.ylabel("MSE Loss")
    plt.title(
        "MSE vs Depth: Quantized vs Full Precision Networks (with Skip Connections)"
    )
    plt.legend()
    plt.grid()
    plt.show()


def train_model(
    target_model, train_model, data, full2quant, epochs=100, lr=0.01, grad_clamp=1
):
    if full2quant:
        lr *= 10
        grad_clamp /= 10
        print("full2quant mode")
    else:
        print("quant2full mode")
    optimizer = optim.Adam(train_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    target_output = target_model(data).detach()
    # print("data", data)
    # print("target_output", target_output)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = train_model(data)
        loss = criterion(output, target_output)
        loss.backward()
        torch.nn.utils.clip_grad_value_(train_model.parameters(), grad_clamp)
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return loss.item()
