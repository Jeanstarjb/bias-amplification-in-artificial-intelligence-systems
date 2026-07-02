import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def measure_bias_amplification(model, data_loader, sensitive_feature_idx, target_idx):
    """
    Measure bias amplification in predictions.
    """
    with torch.no_grad():
        sensitive_feature_values = []
        predictions = []
        targets = []

        for batch in data_loader:
            inputs, labels = batch
            sensitive_feature_values.append(inputs[:, sensitive_feature_idx].numpy())
            predictions.append(torch.argmax(model(inputs), dim=1).numpy())
            targets.append(labels.numpy())

        sensitive_feature_values = np.concatenate(sensitive_feature_values)
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        # Calculate bias amplification
        bias_amplification = {}
        unique_sensitive_values = np.unique(sensitive_feature_values)

        for value in unique_sensitive_values:
            mask = sensitive_feature_values == value
            pred_dist = np.bincount(predictions[mask], minlength=len(np.unique(targets))) / np.sum(mask)
            target_dist = np.bincount(targets[mask], minlength=len(np.unique(targets))) / np.sum(mask)
            amplification = np.abs(pred_dist - target_dist).sum()
            bias_amplification[value] = amplification

        return bias_amplification

if __name__ == '__main__':
    # Dummy dataset
    np.random.seed(42)
    torch.manual_seed(42)

    num_samples = 1000
    input_dim = 5
    hidden_dim = 10
    output_dim = 3

    # Generate synthetic data
    X = np.random.rand(num_samples, input_dim)
    sensitive_feature = np.random.choice([0, 1], size=num_samples)  # Binary sensitive feature
    y = (sensitive_feature + np.random.randint(0, output_dim, size=num_samples)) % output_dim

    # Add sensitive feature as the first column
    X[:, 0] = sensitive_feature

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize and train the model
    model = SimpleClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(10):
        for batch in data_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Measure bias amplification
    sensitive_feature_idx = 0  # Index of the sensitive feature
    target_idx = 1  # Target index (not used directly here)
    bias_amplification = measure_bias_amplification(model, data_loader, sensitive_feature_idx, target_idx)

    print("Bias Amplification per Sensitive Feature Value:")
    for value, amplification in bias_amplification.items():
        print(f"Sensitive Value {value}: Amplification {amplification:.4f}")