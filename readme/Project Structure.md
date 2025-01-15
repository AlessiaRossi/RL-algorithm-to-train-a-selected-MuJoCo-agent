## Project Structure

The project is organized into the following directories and files:

 ```plaintext
 RL-algorithm-to-train-a-selected-MuJoCo-agent/
├── .venv/                     # Virtual Environment
├── functions/                 # Contains the main functions for the project
│    └── ppo.py                # Proximal Policy Optimization (PPO) algorithm
│    └── progressCallback.py   # Callback function for training progress
│    └── record.py             # Record function for saving training data
│    └── sac.py                # Soft Actor-Critic (SAC) algorithm
│    └── utils.py              # Utility functions
├── normalization/             # Contains normalization functions
├── readme/                    # Contains project documentation
│    └── Project Structure.md  # Project structure documentation
│    └── README.md             # Project overview and description
├── results/                   # Contains training results and logs
│    ├── logs/                 # Training logs
│    ├── metrics/              # Metrics files
│    ├── normalization/        # Normalization statistics
│    ├── plots/                # Plots and graphs
│    ├── video/                # Recorded videos of the agent 
├── .gitignore/                # Git ignore file   
├── analysis_results.py/       # Evaluation and analysis script
├── config.py/                 # Configuration file
├── environment.py/            # Environment setup
├── main.py/                   # Main script to run the project
├── requirements.txt/          # Required dependencies

```
### Directory Descriptions

### `main.py`
- **Purpose**: The primary entry point for running the project.
- **Key Functions**:
  - Orchestrates training, evaluation, and debugging workflows.
  - Handles configurations from `config.yaml`.
  - Manages model saving and loading.

### `config.yaml`
- **Purpose**: Configuration file for customizing project parameters.
- **Parameters**:
  - `env_id`: The Gym environment ID (e.g., `HalfCheetah-v5`).
  - `max_episode_steps`: Maximum steps per episode.
  - `n_envs`: Number of environments for vectorized training.
  - `ppo_timesteps`, `sac_timesteps`: Total timesteps for PPO and SAC training.
  - `debug`: Toggle debug mode.
  - Additional hyperparameters for PPO and SAC.

### `requirements.txt`
- **Purpose**: Specifies Python dependencies required to run the project.
- **Usage**: Install dependencies using `pip install -r requirements.txt`.

### `.gitignore`
- **Purpose**: Lists files and directories to be excluded from version control.
- **Examples**:
  - `__pycache__/`
  - `results/`

---


### `functions/`
Contains Python modules with core functionality.

#### `analysis_results.py`
- **Purpose**: Provides evaluation and visualization tools.
- **Key Functions**:
  - `evaluate_model`: Evaluates a trained agent.
  - `plot_comparison`: Generates performance comparison plots for PPO, SAC, and Random policies.
  - `save_metrics`: Saves evaluation metrics to a text file.

#### `environments.py`
- **Purpose**: Defines environment setup utilities.
- **Key Functions**:
  - `make_env`: Initializes a single Gym environment.
  - `make_vec_envs`: Creates vectorized environments.
  - `create_eval_env`: Configures an evaluation environment.

#### `ppo.py`
- **Purpose**: Implements PPO-specific training routines.
- **Key Functions**:
  - `train_ppo`: Trains a PPO model.
  - `hyperparameter_tuning`: Optimizes PPO hyperparameters using Optuna.

#### `sac.py`
- **Purpose**: Implements SAC-specific training routines.
- **Key Functions**:
  - `train_sac`: Trains an SAC model.
  - `hyperparameter_tuning`: Optimizes SAC hyperparameters using Optuna.

#### `record.py`
- **Purpose**: Records video demonstrations of trained agents.
- **Key Functions**:
  - `record_agent_video`: Captures agent performance for visualization.

#### `utils.py`
- **Purpose**: Utility functions for debugging and logging.
- **Key Functions**:
  - `seed_everything`: Sets a global seed for reproducibility.
  - `configure_logger`: Sets up logging for the project.

#### `progressCallback.py`
- **Purpose**: Custom callback for tracking training progress.
- **Key Classes**:
  - `ProgressCallback`: Prints training progress at regular intervals.

---

### `results/`
Stores outputs and logs from the project.

#### `metrics/`
- **Purpose**: Contains evaluation metrics for trained models.
- **Examples**:
  - `random_metrics.txt`: Metrics for the random policy.
  - `ppo_metrics.txt`: Metrics for the PPO agent.
  - `sac_metrics.txt`: Metrics for the SAC agent.

#### `normalization/`
- **Purpose**: Saves normalization statistics for vectorized environments.
- **Examples**:
  - `ppo_vecnormalize_stats.pkl`: Normalization stats for PPO.
  - `sac_vecnormalize_stats.pkl`: Normalization stats for SAC.

#### `plots/`
- **Purpose**: Contains visualizations of evaluation results.
- **Examples**:
  - `performance_comparison.png`: Bar plot comparing rewards.
  - `reward_distribution.png`: Box plot showing reward variability.

#### `videos/`
- **Purpose**: Stores videos of agent performances.
- **Examples**:
  - `ppo_agent/`: Videos for the PPO-trained agent.
  - `sac_agent/`: Videos for the SAC-trained agent.

---