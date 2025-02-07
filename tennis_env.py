import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame

class TennisEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=False):
        super(TennisEnv, self).__init__()
        self.render_mode = render_mode

        # Define action space: 0 = left, 1 = stay, 2 = right.
        self.action_space = spaces.Discrete(3)
        # Observation: [ball.x, ball.y, ball.vx, ball.vy, paddle.x, paddle_vx]
        low = np.array([0, 0, -np.inf, -np.inf, 0, -np.inf], dtype=np.float32)
        high = np.array([800, 400, np.inf, np.inf, 800, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 400))
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        else:
            self.screen = None

        self._setup_world()

    def _setup_world(self):
        # Initialize pymunk space with gravity.
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)
        self._create_boundaries()
        self.ball = self._create_ball()
        self.paddle = self._create_paddle()
        self.last_paddle_x = self.paddle.position.x

    def _create_boundaries(self):
        # Create static walls.
        static_lines = [
            pymunk.Segment(self.space.static_body, (0, 0), (800, 0), 1),    # bottom
            pymunk.Segment(self.space.static_body, (0, 0), (0, 400), 1),    # left
            pymunk.Segment(self.space.static_body, (800, 0), (800, 400), 1),  # right
            pymunk.Segment(self.space.static_body, (0, 400), (800, 400), 1)   # top
        ]
        for line in static_lines:
            line.elasticity = 0.95
        self.space.add(*static_lines)

    def _create_ball(self):
        mass = 1
        radius = 10
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = (400, 100)
        body.velocity = (200, -300)
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 0.9
        shape.collision_type = 1
        self.space.add(body, shape)
        return body

    def _create_paddle(self):
        width, height = 100, 10
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = (400, 350)
        shape = pymunk.Poly.create_box(body, (width, height))
        shape.elasticity = 1.0
        shape.collision_type = 2
        self.space.add(body, shape)
        return body

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        # Remove old objects.
        for obj in [self.ball, self.paddle]:
            for shape in obj.shapes:
                self.space.remove(shape)
            self.space.remove(obj)
        self._setup_world()
        self.episode_hits = 0  # <-- Initialize the hit counter for the episode
        return self._get_obs(), {}

    def _get_obs(self):
        # Calculate paddle velocity.
        current_paddle_x = self.paddle.position.x
        paddle_vx = current_paddle_x - self.last_paddle_x
        self.last_paddle_x = current_paddle_x

        raw_obs = np.array([
            self.ball.position.x,
            self.ball.position.y,
            self.ball.velocity.x,
            self.ball.velocity.y,
            self.paddle.position.x,
            paddle_vx
        ], dtype=np.float32)
        
        # Normalize positions to [-1, 1]
        normalized_obs = raw_obs.copy()
        normalized_obs[0] = (raw_obs[0] / 800.0) * 2 - 1  # ball x
        normalized_obs[1] = (raw_obs[1] / 400.0) * 2 - 1  # ball y
        normalized_obs[4] = (raw_obs[4] / 800.0) * 2 - 1  # paddle x
        # Velocities remain unchanged
        return normalized_obs

    def step(self, action):
        # Update paddle position based on action.
        if action == 0:
            self.paddle.position = (max(50, self.paddle.position.x - 10), self.paddle.position.y)
        elif action == 2:
            self.paddle.position = (min(750, self.paddle.position.x + 10), self.paddle.position.y)
        # No movement for action == 1.

        # Advance the physics simulation.
        self.space.step(1/60.0)

        # Initialize a small time-step penalty to encourage faster responses.
        reward = -0.01
        done = False

        # Check for paddle hit using AABB collision.
        if self._check_paddle_hit():
            self.episode_hits += 1
            # Compute additional bonus based on how centered the ball is relative to the paddle.
            ball_x = self.ball.position.x
            paddle_x = self.paddle.position.x
            dist_error = abs(ball_x - paddle_x)
            bonus = max(0, 1 - (dist_error / 100))  # bonus gets lower when ball is away from center (scale factor: 100)
            # Base reward + bonus for consecutive hits.
            reward = 1.0 + (self.episode_hits * 0.1) + bonus

        # Check termination: if the ball goes below the bottom, consider it a miss.
        if self.ball.position.y > 400:
            done = True
            reward = -1.0
            info = {"episode_hits": self.episode_hits}
            # Log the episode's hit count at termination.
            print(f"Episode ended. Total consecutive hits: {self.episode_hits}")
        else:
            info = {}

        obs = self._get_obs()
        return obs, reward, done, False, info

    def _check_paddle_hit(self):
        ball_x, ball_y = self.ball.position
        paddle_x, paddle_y = self.paddle.position
        paddle_width = 100
        paddle_height = 10
        if (paddle_x - paddle_width/2 <= ball_x <= paddle_x + paddle_width/2 and
            paddle_y - paddle_height/2 <= ball_y <= paddle_y + paddle_height/2):
            return True
        return False

    def render(self, mode="human"):
        if not self.render_mode:
            return None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        self.screen.fill((0, 0, 0))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(60)
        return np.array(pygame.surfarray.array3d(self.screen)).swapaxes(0, 1)

    def close(self):
        if self.render_mode:
            pygame.quit()
