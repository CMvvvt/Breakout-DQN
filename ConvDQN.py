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
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Setup the conv net
        self.conv_net = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, (8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=1),
            nn.ReLU(),
        )
        self.conv_out_dims = self._get_conv_out_dims(state_dim)

        # Setup the fc layer
        self.fc_net = nn.Sequential(
            nn.Linear(self.conv_out_dims, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("mps")
        self.to(self.device)

    # Ensure x is correctly formatted (i.e., [batch_size, channels, height, width]).
    def forward(self, x):
        # x.shape = (32, 4, 84, 84)
        conv_state = self.conv_net(x)
        flat_conv_state = conv_state.view(conv_state.size(0), -1)
        return self.fc_net(flat_conv_state)

    def _get_conv_out_dims(self, state_dim):
        # The dummy tensor should have dimensions [1, channels, height, width]
        dummy_input = torch.zeros(1, *state_dim)

        # Forward the dummy input through the convolutional layers
        with torch.no_grad():
            dummy_output = self.conv_net(dummy_input)

        # Calculate the total number of output features
        return int(torch.prod(torch.tensor(dummy_output.size()[1:])))

    def save_model(self, type, steps, name):
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
