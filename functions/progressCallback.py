from stable_baselines3.common.callbacks import BaseCallback

# A custom callback to print the percentage of training completed at specified intervals
class ProgressCallback(BaseCallback):
    """
    Custom callback for monitoring training progress.
    Prints the percentage of total training timesteps completed at regular intervals.
    """
    def __init__(self, total_timesteps, check_freq=10000, verbose=1):
        """
        Initialize the ProgressCallback.
        Args:
            total_timesteps (int): Total number of timesteps for the training process.
            check_freq (int): Frequency (in steps) at which progress is printed.
            verbose (int): Verbosity level (1 for printing updates, 0 for silent mode).
        """
        super().__init__(verbose)  # Initialize the BaseCallback
        self.total_timesteps = total_timesteps  # Total timesteps for the training
        self.check_freq = check_freq  # Frequency of progress updates

    def _on_step(self) -> bool:
        """
        Method called at each step of the training process.
        Prints the progress if the step count reaches the specified check frequency.
        Returns:
            bool: Always returns True to continue training.
        """
        # Check if the current call count is a multiple of the check frequency
        if self.n_calls % self.check_freq == 0:
            # Calculate the percentage of training completed
            pct = 100.0 * self.n_calls / self.total_timesteps
            # Print the progress
            print(f"Training progress: {pct:.2f}%")
        return True  # Continue training
