import os
import yaml

# Funzione per creare directory
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Funzione per creare un file di configurazione predefinito
def create_default_config(config_path="config.yaml"):
    default_config = {
        "environment": {
            "env_id": "HalfCheetah-v5",
            "max_episode_steps": 1000,
            "n_envs": 4,
            "seed": 42,
        },
        "evaluation": {
            "eval_freq": 50000,
            "n_episodes": 5,
        },
        "ppo": {
            "timesteps": 2000000,
            "training_steps": 50000,
            "n_trials": 20,
        },
        "sac": {
            "timesteps": 2000000,
            "training_steps": 50000,
            "n_trials": 20,
        },
    }
    with open(config_path, "w") as file:
        yaml.safe_dump(default_config, file)
    print(f"File di configurazione creato: {config_path}")


# Funzione per validare la configurazione
def validate_config(config):
    # Controllo dei campi obbligatori e dei valori critici
    required_sections = ["environment", "evaluation", "ppo", "sac"]

    # Verifica che tutte le sezioni siano presenti
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Configurazione non valida. Mancano le seguenti sezioni: {missing_sections}")

    # Controlla i campi nelle rispettive sezioni
    required_keys = {
        "environment": ["env_id", "max_episode_steps", "n_envs", "seed"],
        "evaluation": ["eval_freq", "n_episodes"],
        "ppo": ["timesteps", "training_steps", "n_trials"],
        "sac": ["timesteps", "training_steps", "n_trials"],
    }

    for section, keys in required_keys.items():
        missing_keys = [key for key in keys if key not in config[section] or config[section][key] is None]
        if missing_keys:
            raise ValueError(f"Configurazione non valida. Mancano i seguenti campi in '{section}': {missing_keys}")

    # Controlla i valori critici
    if config["ppo"]["training_steps"] > config["ppo"]["timesteps"]:
        raise ValueError("Training steps di PPO non possono superare i timesteps totali!")
    if config["sac"]["training_steps"] > config["sac"]["timesteps"]:
        raise ValueError("Training steps di SAC non possono superare i timesteps totali!")
    if config["environment"]["n_envs"] <= 0:
        raise ValueError("Il numero di ambienti (n_envs) deve essere maggiore di 0!")
    if config["evaluation"]["n_episodes"] <= 0:
        raise ValueError("Il numero di episodi (n_episodes) deve essere maggiore di 0!")

# Funzione per caricare la configurazione
def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        print(f"File di configurazione non trovato. Creazione di un file predefinito in: {config_path}")
        create_default_config(config_path)
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Convalida della configurazione
    validate_config(config)
    return config
