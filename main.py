import gymnasium as gym
from tennis_env import TennisEnv
from stable_baselines3 import PPO

def run_agent(env, model):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
    env.close()

def main():
    # Create the environment with rendering enabled.
    env = TennisEnv(render_mode=True)
    # Load the saved model.
    model = PPO.load("tennis_ppo_model")
    run_agent(env, model)

if __name__ == "__main__":
    main()
