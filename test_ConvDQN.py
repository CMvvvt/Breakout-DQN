import unittest
import torch
from ConvDQN import ConvDQN


class TestConvDQN(unittest.TestCase):
    def setUp(self):
        self.state_dim = (210, 160, 3)
        self.action_dim = 4
        self.model = ConvDQN(state_dim=self.state_dim, action_dim=self.action_dim)

    def test_forward_output_shape(self):
        """Test the forward pass to ensure it outputs the correct shape"""
        batch_size = 5
        # Creating a dummy input tensor NHWC format
        dummy_input = torch.rand(batch_size, *self.state_dim)
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (batch_size, self.action_dim))

    def test_conv_output_dims(self):
        model = ConvDQN(state_dim=(210, 160, 3), action_dim=4)
        expected_output_dims = 22528  # Calculated manually as shown above
        assert (
            model.conv_out_dims == expected_output_dims
        ), f"Expected {expected_output_dims}, got {model.conv_out_dims}"

    def test_initialization_exception(self):
        """Test whether the model initialization handles incorrect state dimensions"""
        with self.assertRaises(ValueError):
            ConvDQN(
                state_dim=(210, 160), action_dim=self.action_dim
            )  # Incomplete state_dim

    def test_zero_input(self):
        """Test the model with an input of zeros"""
        batch_size = 1
        zero_input = torch.zeros(batch_size, *self.state_dim)
        output = self.model(zero_input)
        self.assertEqual(output.shape, (batch_size, self.action_dim))

    def test_training_step(self):
        """Test that weights change after a training step"""
        batch_size = 2
        dummy_input = torch.rand(batch_size, *self.state_dim)
        dummy_target = torch.randint(0, self.action_dim, (batch_size,))
        output_before = self.model(dummy_input).detach().clone()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        optimizer.zero_grad()
        output = self.model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()

        output_after = self.model(dummy_input).detach()
        # Check if any parameter has changed
        self.assertFalse(
            torch.equal(output_before, output_after),
            "Model parameters did not change after training step.",
        )


if __name__ == "__main__":
    unittest.main()
