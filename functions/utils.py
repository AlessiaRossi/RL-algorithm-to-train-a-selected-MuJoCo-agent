def progressCallback(n_calls, total_timesteps, check_freq=10000, verbose=1):
    """
    Funzione per stampare la % di training completato ogni check_freq step.
    """
    if n_calls % check_freq == 0:
        pct = 100.0 * n_calls / total_timesteps
        if verbose > 0:
            print(f"Training progress: {pct:.2f}%")
    return True
