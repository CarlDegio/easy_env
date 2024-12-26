import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class NavConfig:
    render_mode = "human"

    max_speed = 0.2  # m/s
    max_angular_speed = 2.0  # rad/s
    max_acceleration = 1.0  # m/s^2
    max_angular_acceleration = 2.0  # rad/s^2

    dt = 0.1  # Time step duration
    max_episode_steps = 100


class NavEnv(gym.Env):
    metadata = {"render_fps": 20}

    def __init__(self, config: NavConfig):
        super(NavEnv, self).__init__()

        # 新增 config 变量
        self.config = config
        self.render_mode = config.render_mode
        # Action space: [acceleration a (in forward direction), angular acceleration beta]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -2.0]),
            high=np.array([1.0, 2.0]),
            shape=(2,),
            dtype=np.float32,
        )

        # Observation space: [posx, posy, qx, qy, qz, qw, vx, vy, omega]
        high = np.array([1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
        low = -high
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

        self.dt = self.config.dt  # 使用 config 中的 dt

        self.screen = None
        self.clock = None

        self.current_step = 0
        self.state = None

        # For rendering
        self.trajectory = []

    def reset(self, seed=None, options={'side': None, 'pos': None, 'angle': None}):
        super().reset(seed=seed)
        self.current_step = 0

        # Random initial position on the square's edges

        if options['pos'] is None:
            pos = self.np_random.uniform(-1.0, 1.0)
        else:
            pos = options['pos']

        if options['side'] is None:
            side = self.np_random.choice(["left", "right", "top", "bottom"])
        else:
            side = options['side']

        if side == "left":
            posx = -1.0
            posy = pos
        elif side == "right":
            posx = 1.0
            posy = pos
        elif side == "top":
            posx = pos
            posy = 1.0
        elif side == "bottom":
            posx = pos
            posy = -1.0

        # Random orientation
        if options['angle'] is None:
            angle = self.np_random.uniform(-np.pi, np.pi)
        else:
            angle = options['angle']
        qw = np.cos(angle / 2)
        qz = np.sin(angle / 2)
        qx = 0.0
        qy = 0.0

        # Initial velocities
        vx = 0.0
        vy = 0.0
        omega = 0.0

        self.state = np.array(
            [posx, posy, qx, qy, qz, qw, vx, vy, omega], dtype=np.float32
        )

        self.trajectory = [np.array([posx, posy])]

        if self.render_mode == "human":
            self._render_frame()

        return self.state, {}

    def step(self, action):
        a, beta = action  # Acceleration and angular acceleration
        a = np.clip(a, -self.config.max_acceleration,
                    self.config.max_acceleration)
        beta = np.clip(beta, -self.config.max_angular_acceleration,
                       self.config.max_angular_acceleration)

        # Update dynamics
        self._update_dynamics(a, beta)
        reward = self._get_step_return()

        self.current_step += 1
        truncated = self.current_step >= self.config.max_episode_steps
        terminated = False

        if self.render_mode == "human":
            self._render_frame()

        return self.state, reward, terminated, truncated, {}

    def _get_step_return(self):
        posx, posy, qx, qy, qz, qw, vx, vy, omega = self.state
        distance = np.sqrt(posx**2 + posy**2)
        if abs(posx) <= 0.1 and abs(posy) <= 0.1:
            reward = 0.0
        else:
            reward = -distance
        return reward

    def _update_dynamics(self, a, beta):
        # Unpack state
        posx, posy, qx, qy, qz, qw, vx, vy, omega = self.state

        speed = np.sqrt(vx**2 + vy**2)
        speed += a * self.dt
        speed = np.clip(speed, -self.config.max_speed, self.config.max_speed)
        direction = 2*np.arctan2(qz, qw)
        vx = speed * np.cos(direction)
        vy = speed * np.sin(direction)

        omega += beta * self.dt
        omega = np.clip(omega, -self.config.max_angular_speed,
                        self.config.max_angular_speed)

        # Update positions
        # 只根据当前方向更新位置
        posx += vx * self.dt
        posy += vy * self.dt

        # Update orientation
        delta_angle = omega * self.dt
        delta_qw = np.cos(delta_angle / 2)
        delta_qz = np.sin(delta_angle / 2)
        delta_qx = 0.0
        delta_qy = 0.0

        # Multiply quaternions
        q = np.array([qw, qx, qy, qz])
        delta_q = np.array([delta_qw, delta_qx, delta_qy, delta_qz])
        q_new = self._quaternion_multiply(q, delta_q)
        q_new /= np.linalg.norm(q_new)
        qw, qx, qy, qz = q_new

        self.state = np.array(
            [posx, posy, qx, qy, qz, qw, vx, vy, omega], dtype=np.float32
        )

        self.trajectory.append(np.array([posx, posy]))

    def _quaternion_multiply(self, q1, q2):
        # Quaternion multiplication
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        # Initialize pygame
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((600, 600))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((600, 600))
        self.surf.fill((255, 255, 255))

        # Draw target area
        scale = 200  # Scaling factor
        target_rect = pygame.Rect(
            300 - 0.1 * scale, 300 - 0.1 * scale, 0.2 * scale, 0.2 * scale
        )
        pygame.draw.rect(self.surf, (255, 200, 200), target_rect)

        # Draw trajectory
        for idx, pos in enumerate(self.trajectory):
            posx, posy = pos
            screen_x = 300 + posx * scale
            screen_y = 300 - posy * scale
            color_intensity = min(255, idx * 3)
            pygame.draw.circle(
                self.surf,
                (255 - color_intensity, 0, 0),
                (int(screen_x), int(screen_y)),
                2,
            )

        # Draw robot
        posx, posy, qx, qy, qz, qw, vx, vy, omega = self.state
        screen_x = 300 + posx * scale
        screen_y = 300 - posy * scale

        robot_size = 0.1 * scale
        robot_rect = pygame.Rect(0, 0, robot_size, robot_size)
        robot_rect.center = (screen_x, screen_y)

        # Robot orientation
        angle = 2 * np.arctan2(qz, qw)
        angle_deg = np.degrees(angle)

        # Robot surface
        robot_surf = pygame.Surface((robot_size, robot_size))
        robot_surf.fill((0, 0, 0))
        robot_surf.set_colorkey((0, 0, 0))
        pygame.draw.rect(robot_surf, (0, 255, 0),
                         (0, 0, robot_size, robot_size))

        # Forward direction triangle
        pygame.draw.polygon(
            robot_surf,
            (0, 0, 255),
            [
                (robot_size, robot_size/2),  # 顶点指向前方并向前移动半个车身
                (robot_size/2, 0),
                (robot_size/2, robot_size),
            ],
        )

        # Rotate robot
        rotated_robot = pygame.transform.rotate(robot_surf, angle_deg)
        robot_rect = rotated_robot.get_rect(center=(screen_x, screen_y))

        # Blit to screen
        self.surf.blit(rotated_robot, robot_rect)

        self.screen.blit(self.surf, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    config = NavConfig()
    env = NavEnv(config)
    obs, _ = env.reset(options={'side': 'left', 'pos': 0.0, 'angle': 0.0})

    for _ in range(100):
        action = env.action_space.sample()
        # action[0] = 0.2
        # action[1] = 0.0
        obs, reward, terminated, truncated, info = env.step(action)
        print('reward:', reward)
        if terminated or truncated:
            break

    env.close()
