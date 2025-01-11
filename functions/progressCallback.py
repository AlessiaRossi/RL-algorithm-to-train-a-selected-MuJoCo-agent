from stable_baselines3.common.callbacks import BaseCallback

# Funzione per stampare la % di training completato ogni check_freq step
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            pct = 100.0 * self.n_calls / self.total_timesteps
            print(f"Progresso di training: {pct:.2f}%")
        return True