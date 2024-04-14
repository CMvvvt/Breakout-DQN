import unittest
from unittest.mock import Mock
import numpy as np
import cv2
from utils import ExponentialSchedule, FrameStackingAndResizingEnv


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Create a mock environment
        self.mock_env = Mock()
        self.mock_env.reset.return_value = np.random.randint(
            0, 256, (210, 160, 3), dtype=np.uint8
        )
        self.mock_env.step.return_value = (
            np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8),
            1,
            False,
            {},
        )
        self.mock_env.action_space.sample.return_value = 0

        # Parameters for the frame stacking environment
        self.width = 84
        self.height = 84
        self.num_stack = 4

        # Instantiate the environment wrapper
        self.env = FrameStackingAndResizingEnv(
            self.mock_env, self.width, self.height, self.num_stack
        )

    def test_reset(self):
        """Test that reset returns a correctly shaped and stacked array"""
        initial_state = self.env.reset()
        self.assertEqual(initial_state.shape, (self.num_stack, self.height, self.width))
        # Check if all frames in the stack are the same after reset
        self.assertTrue(np.all(initial_state[0] == initial_state[1]))

    def test_step(self):
        """Test that step updates the frame stack correctly"""
        _ = self.env.reset()
        new_state, reward, done, _ = self.env.step(0)
        self.assertEqual(new_state.shape, (self.num_stack, self.height, self.width))
        # Ensure reward and done are returned correctly
        self.assertEqual(reward, 1)
        self.assertFalse(done)
        # Check that the frame buffer is updated correctly
        self.assertFalse(np.array_equal(new_state[0], new_state[1]))

    def test_preprocess_frame(self):
        """Test frame preprocessing to check resizing and color conversion"""
        frame = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
        processed_frame = self.env._preprocess_frame(frame)
        self.assertEqual(processed_frame.shape, (self.height, self.width))
        # Check if the processed frame is grayscale
        self.assertTrue(len(processed_frame.shape) == 2)

    def test_render(self):
        """Test the render method"""
        _ = self.env.reset()
        frame = self.env.render("rgb_array")
        self.assertEqual(frame.shape, (210, 160, 3))  # Original frame shape
        # Render with a different mode
        with self.assertRaises(AttributeError):
            _ = self.env.render("human")

    def tests_exponential_decay(self):
        _schedule = ExponentialSchedule(0.1, 0.2, 3)
        _test_schedule(_schedule, -1, 0.1)
        _test_schedule(_schedule, 0, 0.1)
        _test_schedule(_schedule, 1, 0.141421356237309515)
        _test_schedule(_schedule, 2, 0.2)
        _test_schedule(_schedule, 3, 0.2)
        del _schedule

        _schedule = ExponentialSchedule(0.5, 0.1, 5)
        _test_schedule(_schedule, -1, 0.5)
        _test_schedule(_schedule, 0, 0.5)
        _test_schedule(_schedule, 1, 0.33437015248821106)
        _test_schedule(_schedule, 2, 0.22360679774997905)
        _test_schedule(_schedule, 3, 0.14953487812212207)
        _test_schedule(_schedule, 4, 0.1)
        _test_schedule(_schedule, 5, 0.1)

        _schedule = ExponentialSchedule(1, 0.01, 1000000)
        print(_schedule.value(950000))
        del _schedule


def _test_schedule(schedule, step, value, ndigits=5):
    """Tests that the schedule returns the correct value."""
    v = schedule.value(step)
    if not round(v, ndigits) == round(value, ndigits):
        raise Exception(
            f"For step {step}, the scheduler returned {v} instead of {value}"
        )


if __name__ == "__main__":
    unittest.main()
