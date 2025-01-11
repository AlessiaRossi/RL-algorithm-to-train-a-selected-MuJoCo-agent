import os
import yaml

# Funzione per creare directory
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Funzione per creare un file di configurazione predefinito
def create_default_config(config_path="config.yaml"):
    default_config = {
        "env_id": "HalfCheetah-v5",
        "max_episode_steps": 1000,
        "seed": 42,
        "eval_freq": 50000,
        "ppo_timesteps": 2000000,
        "ppo_training_steps": 50000,
        "sac_timesteps": 2000000,
        "sac_training_steps": 50000,
        "n_envs": 4,
        "n_episodes": 5,
        "n_trials_ppo": 20,
        "n_trials_sac": 20,
    }
    with open(config_path, "w") as file:
        yaml.safe_dump(default_config, file)
    print(f"File di configurazione creato: {config_path}")

# Funzione per validare la configurazione
def validate_config(config, required_keys):
    missing_keys = [key for key in required_keys if key not in config or config[key] is None]
    if missing_keys:
        raise ValueError(f"Configurazione non valida. Mancano i seguenti campi: {missing_keys}")

    # Verifica dei valori critici
    if config["ppo_training_steps"] > config["ppo_timesteps"]:
        raise ValueError("Training steps di PPO non possono superare i timesteps totali!")
    if config["sac_training_steps"] > config["sac_timesteps"]:
        raise ValueError("Training steps di SAC non possono superare i timesteps totali!")
    if config["n_envs"] <= 0:
        raise ValueError("Il numero di ambienti (n_envs) deve essere maggiore di 0!")
    if config["n_episodes"] <= 0:
        raise ValueError("Il numero di episodi (n_episodes) deve essere maggiore di 0!")

# Funzione per caricare la configurazione
def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        print(f"File di configurazione non trovato. Creazione di un file predefinito in: {config_path}")
        create_default_config(config_path)
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Chiavi richieste nella configurazione
    required_keys = [
        "env_id", "max_episode_steps", "seed", "eval_freq", "ppo_timesteps",
        "ppo_training_steps", "sac_timesteps", "sac_training_steps", "n_envs",
        "n_episodes", "n_trials_ppo", "n_trials_sac"
    ]
    validate_config(config, required_keys)
    return config

# Funzione per stampare la % di training completato ogni check_freq step.
def progressCallback(n_calls, total_timesteps, check_freq=10000, verbose=1):
    if n_calls % check_freq == 0:
        pct = 100.0 * n_calls / total_timesteps
        if verbose > 0:
            print(f"Training progress: {pct:.2f}%")
    return True