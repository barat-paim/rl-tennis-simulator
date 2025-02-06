# Tennis RL Simulation

This project implements a minimal reinforcement learning environment simulating a tennis return scenario. The simulation uses [Pymunk](https://www.pymunk.org) for 2D physics, [Pygame](https://www.pygame.org/) for rendering, and a [Gymnasium](https://gymnasium.farama.org/) environment interface to train an RL agent with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

## Project Structure

- **tennis_env.py**: Defines the Gymnasium environment with realistic ball and paddle physics.
- **train_agent.py**: Trains an RL agent using PPO.
- **main.py**: Runs the trained agent in render mode so you can watch it play.
- **requirements.txt**: Lists all Python dependencies.

## Setup

1. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate    # Linux/Mac
    venv\Scripts\activate       # Windows
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Training the Agent

Run:
```bash
python train_agent.py
