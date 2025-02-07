# **RL-Tennis-Simulator: Reinforcement Learning for Tennis Returns**  

## **🎾 Overview**  
RL-Tennis-Simulator is a **reinforcement learning (RL) environment** designed to train an **AI agent** to return a tennis ball using a paddle. The environment is built using **Gymnasium**, **Pymunk (2D physics),** and **Pygame (rendering)**, with training powered by **Stable-Baselines3 (PPO algorithm).**  

🚀 **Key Features:**  
✅ **Physics-Based RL Environment** → Simulates realistic ball dynamics using Pymunk  
✅ **Gymnasium-Compatible** → Custom RL environment follows OpenAI Gym API standards  
✅ **Trainable RL Agent** → Uses Proximal Policy Optimization (PPO) to learn optimal paddle movements  
✅ **Interactive Visualization** → Watch the agent in action via Pygame rendering  

---

## **🛠️ Tech Stack**  
✅ **Python** – Core programming language  
✅ **Gymnasium** – RL environment interface  
✅ **Stable-Baselines3 (PPO)** – RL algorithm for training  
✅ **Pymunk** – 2D physics engine for ball movement  
✅ **Pygame** – Rendering for interactive visualization  

---

## **📁 Project Structure**  

```bash
├── rl-tennis-simulator/
│   ├── tennis_env.py        # Custom Gymnasium environment with physics simulation
│   ├── train_agent.py       # Training script for PPO RL agent
│   ├── main.py              # Runs the trained agent in real-time
│   ├── requirements.txt     # Python dependencies
│   ├── README.md            # Project documentation (this file)
```

---

## **🚀 Setup & Installation**  

### **1️⃣ Create & Activate Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

---

## **🤖 Training the RL Agent**  

### **1️⃣ Run Training Script (Using PPO Algorithm)**  
```bash
python train_agent.py
```
This trains the agent using **10,000 timesteps**. Adjust `total_timesteps` in `train_agent.py` for longer training.  

### **2️⃣ Save & Load the Model**  
```python
from stable_baselines3 import PPO

# Save the trained model
model.save("tennis_ppo_model")

# Load the model for inference
model = PPO.load("tennis_ppo_model")
```

---

## **🎾 Running the Trained Agent**  

Once the agent is trained, watch it play in real-time with **Pygame rendering**:  
```bash
python main.py
```

---

## **📊 Environment Details**  

### **State Space (Observations)**  
Each state consists of **6 features**:  
1. **Ball X position** (float)  
2. **Ball Y position** (float)  
3. **Ball X velocity** (float)  
4. **Ball Y velocity** (float)  
5. **Paddle X position** (float)  
6. **Paddle X velocity** (float)  

### **Action Space**  
The RL agent can take **3 discrete actions**:  
- **0** → Move paddle **left**  
- **1** → **Stay** in the same position  
- **2** → Move paddle **right**  

### **Rewards Shaping**  
- **+1 reward** → When the agent successfully returns the ball, with an additional bonus based on how centered the ball is on the paddle and an incremental bonus for consecutive hits.  
- **-0.01 penalty** → Applied every step to encourage faster responses.  
- **-1 penalty** → When the ball falls past the paddle, the episode ends with a reward of -1.0, and the total `episode_hits` is passed in the info dictionary.  

### **Metrics & Logs**
- **episode_hits** → Total number of successful hits in an episode.
- **episode_hits_rate** → Ratio of successful hits to total attempts.
- **episode_duration** → Number of steps taken in an episode.
- **episode_reward** → Total reward accumulated in an episode.

---

## **🖼️ Visualizing the Simulation**  
The environment renders a **2D simulation** of the paddle and ball using **Pygame**.  

✅ **White Paddle** → Agent-controlled  
✅ **Yellow Ball** → Follows physics-based movement  
✅ **Black Background** → Simplifies visualization  

---

## **🔮 Next Steps & Improvements**  
🔹 **Train with More Advanced RL Algorithms** (e.g., SAC, DDPG)  
🔹 **Enhance Reward Function** (Improve paddle precision feedback)  
🔹 **Multi-Agent Training** (Opponent AI for rallies)  
🔹 **Improve Rendering** (More realistic visuals)  

---

## **📌 Why This Project Matters?**  
RL-Tennis-Simulator **demonstrates reinforcement learning in a physics-based setting**, providing:  
- **Hands-on RL environment development experience**  
- **Practical application of Gymnasium & Stable-Baselines3**  
- **A foundation for AI-based sports simulations**  