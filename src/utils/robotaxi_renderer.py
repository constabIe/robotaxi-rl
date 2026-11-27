"""Pygame front-end for RobotaxiEnv: car-like keyboard control, HUD and video recording."""

import math

import imageio.v2 as imageio
import jax.numpy as jnp
import numpy as np
import pygame

from utils.renderer import BLACK, BLUE, DARK_GRAY, GREEN, RED, WHITE, YELLOW, PygameFrontend

CAR_LENGTH_M = 4.5
CAR_WIDTH_M = 1.8

HUD_KEYS = (
    ("time", "{:d}"),
    ("speed_kmh", "{:.1f} km/h"),
    ("remaining_m", "{:.0f} m"),
    ("distance_to_path", "{:.2f} cells"),
    ("r_progress", "{:+.3f}"),
)


class RobotaxiFrontend(PygameFrontend):
    """Keyboard: up/W throttle, down/S brake, left/A and right/D steer.

    With ``record_path`` set, every frame is stored and written as a video when
    the window closes or ``max_episodes`` episodes have finished.
    """

    def __init__(
        self,
        env,
        env_params,
        init_state,
        eval_mode=False,
        agent_fn=None,
        record_path=None,
        max_episodes=None,
        realtime=True,
    ):
        super().__init__(env, env_params, init_state, eval_mode=eval_mode, agent_fn=agent_fn)
        pygame.display.set_caption("Robotaxi")
        self.record_path = record_path
        self.max_episodes = max_episodes
        self.realtime = realtime
        self.decision_rate = env_params.fps / env_params.action_repeat
        self.frames: list[np.ndarray] = []
        self.episode_return = 0.0
        self.last_return = 0.0
        self.episodes = 0

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        accel = float(keys[pygame.K_UP] or keys[pygame.K_w]) - float(keys[pygame.K_DOWN] or keys[pygame.K_s])
        steer = float(keys[pygame.K_LEFT] or keys[pygame.K_a]) - float(keys[pygame.K_RIGHT] or keys[pygame.K_d])
        return jnp.array([accel, steer], dtype=jnp.float32)

    def world_to_px(self, x: float, y: float) -> tuple[int, int]:
        """World (x right, y up, cells) -> screen pixels (cell centres at integer coordinates)."""
        h = int(self.params.map_height_width[0])
        return int((x + 0.5) * self.CELL_SIZE), int((h - 1 - y + 0.5) * self.CELL_SIZE)

    def draw_info(self, screen, info, position=(8, 8), font_size=14, line_height=18, color=BLACK):
        font = pygame.font.SysFont("Arial", font_size)
        lines = [f"{k}: {fmt.format(int(info[k]) if fmt == '{:d}' else float(info[k]))}" for k, fmt in HUD_KEYS]
        lines += [f"return: {self.episode_return:+.1f}", f"last return: {self.last_return:+.1f}"]
        panel = pygame.Surface((190, line_height * len(lines) + 8), pygame.SRCALPHA)
        panel.fill((255, 255, 255, 215))
        screen.blit(panel, position)
        x, y = position[0] + 6, position[1] + 4
        for line in lines:
            screen.blit(font.render(line, True, color), (x, y))
            y += line_height

    def draw_car(self):
        s, scale = self.state, self.params.meters_per_cell
        x, y, yaw = float(s.agent_pos[0]), float(s.agent_pos[1]), float(s.yaw)
        c, sn = math.cos(yaw), math.sin(yaw)
        hl, hw = CAR_LENGTH_M / (2 * scale), CAR_WIDTH_M / (2 * scale)
        corners = [(hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)]
        pts = [self.world_to_px(x + c * a - sn * b, y + sn * a + c * b) for a, b in corners]
        pygame.draw.polygon(self.screen, BLUE, pts)
        nose = self.world_to_px(x + c * hl, y + sn * hl)
        pygame.draw.line(self.screen, YELLOW, self.world_to_px(x, y), nose, 2)

    def draw(self):
        self.screen.fill(WHITE)
        s = self.state
        path = self.rotate_to_map_vmap(s.path_array)
        self.draw_path(path, self.rotate_to_map_vmap(jnp.atleast_2d(s.agent_pos))[0])
        for obstacles, colour in ((s.kinematic_obstacles, RED), (s.static_obstacles, DARK_GRAY)):
            for ox, oy in np.asarray(obstacles):
                self.draw_square_with_striped_circle(
                    self.world_to_px(ox, oy), self.OBSTACLE_SIDE_PX, self.COLLIDER_RADIUS_PX, colour
                )
        pygame.draw.circle(self.screen, GREEN, self.world_to_px(*np.asarray(s.goal_pos)), self.GOAL_RADIUS_PX)
        rays = jnp.flip(self.obs.collision_rays, axis=-1) * jnp.array([-1, 1])[None, :]
        agent_map = self.rotate_to_map_vmap(jnp.atleast_2d(s.agent_pos))[0]
        self.draw_rays_transparent(self.screen, agent_map, rays, self.CELL_SIZE)
        self.draw_car()
        self.draw_info(self.screen, self.info)
        pygame.display.flip()
        if self.record_path is not None:
            frame = pygame.surfarray.array3d(self.screen)  # (W, H, 3)
            self.frames.append(np.transpose(frame, (1, 0, 2)))

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            action = self.agent_fn(self.state) if self.eval_mode and self.agent_fn else self.handle_keys()
            self.obs, self.state, reward, done, self.info = self.env.step(
                self.key, self.state, action, self.params
            )
            self.episode_return += float(reward)
            self.draw()

            if bool(done):
                self.last_return, self.episode_return = self.episode_return, 0.0
                self.episodes += 1
                if self.max_episodes is not None and self.episodes >= self.max_episodes:
                    self.running = False
            if self.realtime:
                self.clock.tick(self.decision_rate)

        if self.record_path is not None and self.frames:
            imageio.mimsave(self.record_path, self.frames, fps=math.ceil(self.decision_rate), macro_block_size=8)
            print(f"saved {len(self.frames)} frames to {self.record_path}")
        pygame.quit()
