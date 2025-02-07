import gymnasium as gym
from tennis_env import TennisEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

def main():
    # Create the environment (without rendering for faster training) and wrap it with a Monitor.
    env = Monitor(TennisEnv(render_mode=False))
    # Optional: check that the environment follows Gymnasium's API.
    check_env(env, warn=True)

    # Create the PPO agent with a TensorBoard logger.
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tennis_tensorboard/")

    # Define a custom callback to log extra metrics (episode_hits) to TensorBoard.
    class CustomLoggingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            # The Monitor file wrapper logs 'infos' in self.locals.
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode_hits" in info:
                    self.logger.record("episode/total_hits", info["episode_hits"])
            return True

    callback = CustomLoggingCallback()

    # Train the agent. Adjust total_timesteps as needed for convergence.
    model.learn(total_timesteps=50000, callback=callback)
    # Save the trained model.
    model.save("tennis_ppo_model")
    env.close()

if __name__ == "__main__":
    main()
