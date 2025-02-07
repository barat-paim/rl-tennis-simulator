import gymnasium as gym
from tennis_env import TennisEnv
from stable_baselines3 import PPO

def run_agent(env, model):
    total_hits = 0
    num_episodes = 10  # Test across multiple episodes
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_hits = 0
        done = False
        
        while not done:
            # Allow some exploration during evaluation
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, _, info = env.step(action)
            
            if done and "episode_hits" in info:
                episode_hits = info["episode_hits"]
            
            env.render()
        
        total_hits += episode_hits
        print(f"Episode ended with {episode_hits} hits")
    
    print(f"\nAverage hits per episode: {total_hits/num_episodes:.1f}")
    env.close()

def main():
    # Create the environment with rendering enabled.
    env = TennisEnv(render_mode=True)
    # Load the saved model.
    model = PPO.load("tennis_ppo_model")
    run_agent(env, model)

if __name__ == "__main__":
    main()
