import os
import yaml


# Function to ensure the existence of a directory
def ensure_dir(directory):
    """
    Ensures that the specified directory exists. If not, it creates it.
    Args:
        directory (str): Path of the directory to ensure.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


# Function to create a default configuration file
def create_default_config(config_path="config.yaml"):
    """
    Creates a default YAML configuration file for the RL training setup.
    Args:
        config_path (str): Path to save the configuration file.
    """
    default_config = {
        "environment": {
            "env_id": "HalfCheetah-v5",  # Environment ID to use
            "max_episode_steps": 1000,  # Maximum steps per episode
            "n_envs": 4,  # Number of parallel environments
            "seed": 42,  # Seed for reproducibility
        },
        "evaluation": {
            "eval_freq": 50000,  # Frequency of evaluations
            "n_episodes": 5,  # Number of episodes per evaluation
        },
        "ppo": {
            "timesteps": 2000000,  # Total training timesteps for PPO
            "training_steps": 50000,  # Training steps for each update
            "n_trials": 20,  # Number of trials for hyperparameter tuning
        },
        "sac": {
            "timesteps": 2000000,  # Total training timesteps for SAC
            "training_steps": 50000,  # Training steps for each update
            "n_trials": 20,  # Number of trials for hyperparameter tuning
        },
        "general": {
            "debug": False,  # Debug mode flag
            "debug_paths": {  # Paths to saved models and stats
                "ppo_model": "results/logs/ppo/best_model.zip",
                "sac_model": "results/logs/sac/best_model.zip",
                "ppo_stats": "results/normalization/ppo_vecnormalize_stats.pkl",
                "sac_stats": "results/normalization/sac_vecnormalize_stats.pkl",
            }
        }
    }

    # Save the default configuration to a YAML file
    with open(config_path, "w") as file:
        yaml.safe_dump(default_config, file)
    print(f"Configuration file created: {config_path}")


# Function to validate the configuration file
def validate_config(config):
    """
    Validates the configuration file, ensuring all required fields and valid values are present.
    Args:
        config (dict): The configuration dictionary to validate.
    Raises:
        ValueError: If any required fields or values are missing or invalid.
    """
    # Required sections for the configuration
    required_sections = ["environment", "evaluation", "ppo", "sac"]

    # Ensure all required sections are present
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Invalid configuration. Missing sections: {missing_sections}")

    # Required keys for each section
    required_keys = {
        "environment": ["env_id", "max_episode_steps", "n_envs", "seed"],
        "evaluation": ["eval_freq", "n_episodes"],
        "ppo": ["timesteps", "training_steps", "n_trials"],
        "sac": ["timesteps", "training_steps", "n_trials"],
        "general": ["debug", "debug_paths"],
    }

    # Validate each section for missing keys
    for section, keys in required_keys.items():
        missing_keys = [key for key in keys if key not in config[section] or config[section][key] is None]
        if missing_keys:
            raise ValueError(f"Invalid configuration. Missing keys in '{section}': {missing_keys}")

    # Validate critical values
    if config["ppo"]["training_steps"] > config["ppo"]["timesteps"]:
        raise ValueError("PPO training steps cannot exceed total timesteps!")
    if config["sac"]["training_steps"] > config["sac"]["timesteps"]:
        raise ValueError("SAC training steps cannot exceed total timesteps!")
    if config["environment"]["n_envs"] <= 0:
        raise ValueError("Number of environments (n_envs) must be greater than 0!")
    if config["evaluation"]["n_episodes"] <= 0:
        raise ValueError("Number of evaluation episodes (n_episodes) must be greater than 0!")


# Function to load the configuration file
def load_config(config_path="config.yaml"):
    """
    Loads the configuration file. If it doesn't exist, creates a default configuration file.
    Args:
        config_path (str): Path to the configuration file.
    Returns:
        dict: Loaded configuration dictionary.
    """
    # Check if the configuration file exists
    if not os.path.exists(config_path):
        print(f"Configuration file not found. Creating a default file at: {config_path}")
        create_default_config(config_path)

    # Load the YAML configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Validate the configuration
    validate_config(config)
    return config
