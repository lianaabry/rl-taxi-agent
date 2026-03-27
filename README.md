🚕 Reinforcement Learning Taxi Agent (Q-Learning)

 Project Overview

This project demonstrates a Reinforcement Learning (RL) agent trained to solve the Taxi-v3 environment using Q-learning.
The agent learns through trial and error to:

- Pick up a passenger
- Navigate the grid world
- Drop off the passenger at the correct destination

This project uses the Gymnasium and implements a Q-table-based learning algorithm.

🎯 Goal
Train an AI agent that:

✔ Maximizes total reward
✔ Learns optimal actions
✔ Solves the Taxi problem efficiently

🧠 Key Concepts Used
- Reinforcement Learning
- Q-Learning
- Exploration vs Exploitation
- Markov Decision Process (MDP)
- Reward optimization

📦 Environment
The project uses:
env = gym.make("Taxi-v3")

This is a classic RL environment from Gymnasium.

Environment Features:
- 500 possible states
- 6 possible actions

Reward system:

+20 → successful dropoff
-10 → illegal action
-1 → each step

📁 Project Structure
rl-taxi-agent/
│
├── train_agent.py        # Train the Q-learning agent
├── evaluate_agent.py     # Test the trained agent
├── q_table.npy           # Saved trained model
├── requirements.txt      # Dependencies
└── README.md

⚙️ Installation
1. Clone the repository
git clone https://github.com/yourusername/rl-taxi-agent.git
cd rl-taxi-agent
2. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
3. Install dependencies
pip install -r requirements.txt

🚀 Training the Agent
python train_agent.py

This will:
- Initialize a Q-table
- Train using Q-learning
- Save the model as q_table.npy
▶️ Running the Trained Agent
python evaluate_agent.py

This will:
- Load the trained Q-table
- Run the agent in the environment
- Display the taxi solving the task

🧮 Q-Learning Formula

Q(s,a) = (1 - α) * Q(s,a) + α * (reward + γ * max Q(s’,a’))

Where:
α → learning rate
γ → discount factor
Q → Q-table

🔧 Hyperparameters

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 10000

📊 Results

The agent learns optimal policies over time
Exploration decreases as training progresses
Achieves stable performance after training

📈 Future Improvements

Implement Deep Q Network (DQN) using neural networks
Add training visualization (reward graph)
Compare Q-learning vs SARSA
Apply RL to more complex environments
Build a GUI or animation for visualization