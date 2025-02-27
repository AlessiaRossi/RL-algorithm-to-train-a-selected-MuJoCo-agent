# **_RL-algorithm-to-train-a-selected-MuJoCo-agent_**

## Index

1. [Project Description](#1-project-description)
   - [Project Structure.md](readme/Project Structure.md)
2. [Environment setup](#2-environment-setup)
3. [Pipeline](#3-pipeline)
4. [Results & Graphs](#4-results--graphs)
   - [Results & Graphs.md](../results/Results%20%26%20Graphs.md)


## 1. Project Description
This project, made by Alessia Rossi and Gabriele Peluzzi, aims to implement Reinforcement Learning (RL) algorithms to train the HalfCheetah agent from the Gymnasium environments. The main focus is to develop and evaluate RL algorithms, specifically PPO (Proximal Policy Optimization) and SAC (Soft Actor-Critic), to effectively train the agent to perform specific tasks within the MuJoCo environment.

### Reinforcement Learning
In this project, we use advanced RL techniques to train the HalfCheetah agent. The approach involves defining the environment, setting up the agent, and implementing the training algorithms. By analyzing the performance of the agent, we can fine-tune the algorithms to achieve optimal results.

#### Proximal Policy Optimization (PPO)
PPO is a policy gradient method for RL. It uses a surrogate objective function to enable multiple epochs of mini-batch updates. PPO strikes a balance between performance and ease of implementation, making it a popular choice for many RL tasks.

#### Soft Actor-Critic (SAC)
SAC is an off-policy actor-critic algorithm that aims to maximize both the expected reward and the entropy of the policy. This encourages exploration and robustness to changes in the environment. SAC is known for its stability and efficiency in training complex agents.

To recap:
- **Objective**: Implement and evaluate RL algorithms to train the HalfCheetah agent.
- **Key Focus**: Develop effective training algorithms for the agent.
- **Approach**:
  - Define the environment and agent.
  - Implement the PPO and SAC training algorithms.
  - Evaluate and perform hyperparameter tuning based on the agent's performance.
- **Methods**:
  - Use advanced RL techniques and algorithms to train the agent.

[Quickly return to the top](#rl-algorithm-to-train-a-selected-mujoco-agent)

## **2. Environment setup**
Before running the code, ensure you are using Python version 3.10.*. It's important to take some precautions and set up your environment properly. Follow these steps:
1. Create a Virtual Environment:
   - Open your terminal or command prompt.
   - Run the following command to create a virtual environment named "venv": `python -m venv venv`
2. Activate the Virtual Environment:
   - If you're using Windows: `.\venv\Scripts\activate`
   - If you're using Unix or MacOS: `source ./venv/Scripts/activate`
3. OPTIONAL - Deactivate the Environment (When Finished):
   - Use the following command to deactivate the virtual environment: `deactivate`
4. Install Dependencies:
   - After cloning the project and activating the virtual environment, install the required dependencies using: `pip install -r requirements.txt`
     This command downloads all the non-standard modules required by the application.
5. If your Python version used to generate the virtual environment doesn't contain an updated version of pip, update pip using: `pip install --upgrade pip`

Once you've set up your virtual environment and installed the dependencies, you're ready to run the application. Simply navigate to the [`main.py`](main.py) file and execute it.

### Debug Mode
To avoid running the full training and hyperparameter tuning process every time, which is very long and computationally intensive, you can enable the debug mode. This mode allows you to quickly test and debug the code without performing the complete training.

To enable debug mode, set the `debug` parameter to `true` in the `config.yaml` file. This will use pre-trained models and saved statistics for quick testing.

```yaml
general:
  debug: true                  # Enable debug mode
  debug_paths:
    ppo_model: "results/logs/ppo/best_model.zip"
    ppo_stats: "results/normalization/ppo_vecnormalize_stats.pkl"
    sac_model: "results/logs/sac/best_model.zip"
    sac_stats: "results/normalization/sac_vecnormalize_stats.pkl"
```
[Quickly return to the top](#rl-algorithm-to-train-a-selected-mujoco-agent) 

## **3. Pipeline**
The data processing and analysis pipeline includes the following steps:
1. **Environment Setup**: Define and initialize the MuJoCo environment.
2. **Agent Setup**: Configure the RL agent and its parameters.
3. **Training**: Implement the PPO and SAC training algorithms to train the agent.
4. **Evaluation**: Evaluate the performance of the agent and perform hyperparameter tuning.
5. **Analysis**: Analyze the results and performance metrics to determine the effectiveness of the training algorithms.

You can find more details in the _Pipeline Description_ paragraph of the [Project Structure.md](readme/Project Structure.md) file.

[Quickly return to the top](#rl-algorithm-to-train-a-selected-mujoco-agent)

## **4. Results & Graphs**
The goal of the challenge is to train the HalfCheetah agent using PPO and SAC algorithms and evaluate their performance. This involves analyzing the agent's behavior and performance metrics to determine which algorithm is better suited for the problem. Additionally, an analysis of hyperparameters will be conducted to identify which settings yield the best results.

For a detailed visualization of the results and graphs obtained, please refer to the [Results & Graphs.md](../results/Results%20%26%20Graphs.md) file.

[Quickly return to the top](#rl-algorithm-to-train-a-selected-mujoco-agent)