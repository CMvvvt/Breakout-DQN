import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os


class ConvDQN(nn.Module):
    def __init__(self, state_dim, action_dim, lr=0.0025):
        super(ConvDQN, self).__init__()

        # Save directory
        self.save_dir = "models/"
        # (4, 84, 84)
        self.action_dim = action_dim

        # Setup the conv net
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Setup the fc layer
        self.fc_net = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("mps")
        self.to(self.device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            m.bias.data.fill_(0.0)

        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    # Ensure x is correctly formatted (i.e., [batch_size, channels, height, width]).
    def forward(self, x):
        x = x.to(self.device).float() / 255.0

        # print("X's shape:", x.shape)
        # [32, 4, 84,84]
        x = self.conv_net(x)
        x = self.fc_net(x.view(x.size(0), -1))

        return x

    def save_model(self, type, name, steps):
        print(f"Saving {type} model...")
        directory = os.path.join(self.save_dir, name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        path = os.path.join(directory, f"{type}_{str(steps)}.pt")
        torch.save(self.state_dict(), path)

    def load_model(self, type, steps, name):
        print(f"Loading {type} model...")
        path = os.path.join(self.save_dir, name, f"{type}_{str(steps)}.pt")
        if os.path.isfile(path):
            self.load_state_dict(torch.load(path))
        else:
            print("Model file not found.")


if __name__ == "__main__":
    test_conv_output_dims()
