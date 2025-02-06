import gymnasium as gym
from tennis_env import TennisEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def main():
    # Create the environment (without rendering for faster training).
    env = TennisEnv(render_mode=False)
    # Optional: check that the environment follows Gymnasium's API.
    check_env(env, warn=True)

    # Create the PPO agent.
    model = PPO("MlpPolicy", env, verbose=1)
    # Train the agent. Adjust total_timesteps as needed.
    model.learn(total_timesteps=10000)
    # Save the trained model.
    model.save("tennis_ppo_model")
    env.close()

if __name__ == "__main__":
    main()
