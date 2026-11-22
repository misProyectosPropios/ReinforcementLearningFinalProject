"""
Entorno PacmanEnv (Gymnasium) con soporte para múltiples fantasmas, donde cada
fantasma tiene su propia política de movimiento (inspiradas en Pac-Man clásico).

Reglas del juego:
- 1 Pac-Man y N fantasmas (`G` en el mapa, al menos 1).
- Cada fantasma tiene su propia política (se puede configurar con `ghost_policies`
  o se asignan automáticamente en orden a cada fantasma del mapa):
    1. `"chaser"` (Blinky - rojo): persigue a Pac-Man por el camino más corto con
       probabilidad `ghost_chase_prob` (y al azar el resto de las veces).
    2. `"ambusher"` (Pinky - rosado): apunta 3 celdas por delante de la dirección
       actual de Pac-Man para emboscarlo.
    3. `"patrol"` (Inky - celeste): si está lejos de Pac-Man (> 4 pasos) lo persigue,
       pero si está cerca patrulla hacia la esquina opuesta.
    4. `"random"` (Clyde - naranja): se mueve de manera completamente aleatoria
       (evitando volver hacia atrás si tiene otra salida).
    También se puede pasar una función `callable(env, ghost_idx, vecinos) -> (r, c)`
    o un número `float` entre 0 y 1 (probabilidad de persecución individual).
- El episodio termina si: Pac-Man se come todas las bolitas (gana), algún fantasma
  lo atrapa (pierde) o se llega a `max_steps` (truncado).
- Acciones: 0 = arriba, 1 = abajo, 2 = izquierda, 3 = derecha.
- Observación: 4 canales aplanados (paredes, bolitas, Pac-Man, fantasmas), valores 0/1.
"""

from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

from maps import MAPA, MAPA_MULTI_GHOST


class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # arriba, abajo, izquierda, derecha
    DEFAULT_POLICIES = ["chaser", "ambusher", "patrol", "random"]
    GHOST_COLORS = [
        (255, 40, 40),    # Fantasma 0 (Chaser / Blinky): Rojo
        (255, 150, 210),  # Fantasma 1 (Ambusher / Pinky): Rosado
        (0, 230, 230),    # Fantasma 2 (Patrol / Inky): Celeste
        (255, 165, 0),    # Fantasma 3 (Random / Clyde): Naranja
    ]

    def __init__(
        self,
        mapa,
        max_steps=200,
        ghost_chase_prob=0.7,
        ghost_policies=None,
        full_info_obs=False,
        render_mode=None,
        r_pellet=1.0,
        r_step=-0.05,
        r_wall=-0.1,
        r_win=10.0,
        r_dead=-10.0,
        r_win_remaining_coef=0.1,
        r_closer_pellet=0.0,
        r_farther_ghost=0.0,
        r_progress_coef=0.0,
    ):
        super().__init__()
        self.custom_ghost_policies = ghost_policies
        self.full_info_obs = full_info_obs
        self.max_steps = max_steps
        self.ghost_chase_prob = ghost_chase_prob
        self.render_mode = render_mode
        self.r_pellet, self.r_step, self.r_wall = r_pellet, r_step, r_wall
        self.r_win, self.r_dead = r_win, r_dead
        self.r_win_remaining_coef = r_win_remaining_coef
        self.r_closer_pellet = r_closer_pellet
        self.r_farther_ghost = r_farther_ghost
        self.r_progress_coef = r_progress_coef

        self._parse_and_init_map(mapa)

        self.action_space = spaces.Discrete(4)
        self.num_channels = 8 if self.full_info_obs else 4
        self.extra_features_dim = 10 if self.full_info_obs else 0
        obs_dim = self.num_channels * self.H * self.W + self.extra_features_dim
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(obs_dim,), dtype=np.float32
        )

    def _parse_and_init_map(self, mapa):
        filas = [list(f) for f in mapa]
        if len({len(f) for f in filas}) != 1:
            raise ValueError("Todas las filas del mapa deben tener el mismo largo.")
        grid = np.array(filas)
        if (grid == "P").sum() != 1:
            raise ValueError("El mapa debe tener exactamente una 'P'.")
        if (grid == "G").sum() < 1:
            raise ValueError("El mapa debe tener al menos una 'G'.")
        if (grid == ".").sum() == 0:
            raise ValueError("El mapa necesita al menos una bolita '.'.")

        self.H, self.W = grid.shape
        self.walls = grid == "#"
        self.pellets_init = grid == "."
        self.total_pellets_init = max(1, int(self.pellets_init.sum()))
        self.pac_start = tuple(int(x) for x in np.argwhere(grid == "P")[0])
        self.ghost_starts = [
            tuple(int(x) for x in pos) for pos in np.argwhere(grid == "G")
        ]
        self.ghost_policies = self._resolve_policies(len(self.ghost_starts))

    def _resolve_policies(self, num_ghosts):
        if self.custom_ghost_policies is None:
            return [
                self.DEFAULT_POLICIES[i % len(self.DEFAULT_POLICIES)]
                for i in range(num_ghosts)
            ]
        if isinstance(self.custom_ghost_policies, (str, float, int)) or callable(
            self.custom_ghost_policies
        ):
            return [self.custom_ghost_policies for _ in range(num_ghosts)]
        return [
            self.custom_ghost_policies[i % len(self.custom_ghost_policies)]
            for i in range(num_ghosts)
        ]

    @property
    def ghost(self):
        """Compatibilidad hacia atrás: devuelve la posición del primer fantasma."""
        return self.ghosts[0]

    @ghost.setter
    def ghost(self, val):
        self.ghosts[0] = val

    @property
    def ghost_start(self):
        """Compatibilidad hacia atrás: devuelve el inicio del primer fantasma."""
        return self.ghost_starts[0]

    # ---------- utilidades ----------
    def _free(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W and not self.walls[r, c]

    def _obs(self):
        pac = np.zeros((self.H, self.W), np.float32)
        pac[self.pac] = 1.0
        gh = np.zeros((self.H, self.W), np.float32)
        for g_pos in self.ghosts:
            gh[g_pos] = 1.0

        base_channels = [
            self.walls.astype(np.float32).ravel(),
            self.pellets.astype(np.float32).ravel(),
            pac.ravel(),
            gh.ravel(),
        ]
        if not self.full_info_obs:
            return np.concatenate(base_channels)

        # Información completa del entorno (8 canales + 10 variables escalares de estado):
        # Canal 4..7: posición individual de cada fantasma por política (chaser, ambusher, patrol, random/otros)
        ind_ghosts = [np.zeros((self.H, self.W), np.float32) for _ in range(4)]
        for idx, g_pos in enumerate(self.ghosts):
            ind_ghosts[idx % 4][g_pos] = 1.0

        # Variables globales normalizadas en [0.0, 1.0]
        dist_from_pac = self._bfs(self.pac)
        max_possible_dist = float(self.H * self.W)

        # Distancias BFS hacia la bolita más cercana y hacia el fantasma más cercano
        pellet_positions = np.argwhere(self.pellets)
        if len(pellet_positions) > 0:
            min_pellet_dist = min(
                float(dist_from_pac[tuple(p)]) for p in pellet_positions
            )
            if np.isinf(min_pellet_dist):
                min_pellet_dist = max_possible_dist
        else:
            min_pellet_dist = 0.0

        min_ghost_dist = min(
            (float(dist_from_pac[g]) for g in self.ghosts), default=max_possible_dist
        )
        if np.isinf(min_ghost_dist):
            min_ghost_dist = max_possible_dist

        # Movimientos válidos desde la celda actual (arriba, abajo, izquierda, derecha)
        valid_moves = [
            1.0 if self._free(self.pac[0] + dr, self.pac[1] + dc) else 0.0
            for dr, dc in self.MOVES
        ]

        # Dirección actual de Pac-Man codificada en [0, 1]
        dir_up = 1.0 if self.pac_dir == (-1, 0) else 0.0
        dir_down = 1.0 if self.pac_dir == (1, 0) else 0.0
        dir_left = 1.0 if self.pac_dir == (0, -1) else 0.0
        dir_right = 1.0 if self.pac_dir == (0, 1) else 0.0

        remaining_time_ratio = float(max(0, self.max_steps - self.t)) / float(
            max(1, self.max_steps)
        )
        remaining_pellets_ratio = float(self.pellets.sum()) / float(
            self.total_pellets_init
        )

        extra_features = np.array(
            [
                remaining_time_ratio,
                remaining_pellets_ratio,
                min_pellet_dist / max_possible_dist,
                min_ghost_dist / max_possible_dist,
                *valid_moves,
                0.5 * dir_up + 1.0 * dir_down,
                0.5 * dir_left + 1.0 * dir_right,
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                *base_channels,
                *(g_map.ravel() for g_map in ind_ghosts),
                extra_features,
            ]
        )

    def _bfs(self, origen):
        """Distancia (en pasos) desde `origen` a cada celda alcanzable."""
        dist = np.full((self.H, self.W), np.inf)
        dist[origen] = 0
        q = deque([origen])
        while q:
            r, c = q.popleft()
            for dr, dc in self.MOVES:
                nr, nc = r + dr, c + dc
                if self._free(nr, nc) and dist[nr, nc] == np.inf:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
        return dist

    def _nearest_free_cell(self, target_r, target_c):
        """Encuentra la celda libre más cercana a `(target_r, target_c)`."""
        target_r = int(np.clip(target_r, 0, self.H - 1))
        target_c = int(np.clip(target_c, 0, self.W - 1))
        if self._free(target_r, target_c):
            return (target_r, target_c)

        visited = {(target_r, target_c)}
        q = deque([(target_r, target_c)])
        while q:
            r, c = q.popleft()
            for dr, dc in self.MOVES:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W and (nr, nc) not in visited:
                    if self._free(nr, nc):
                        return (nr, nc)
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return self.pac

    def _step_towards(self, vecinos, target):
        dist = self._bfs(target)
        mejor = min(dist[v] for v in vecinos)
        return [v for v in vecinos if dist[v] == mejor]

    def _select_ghost_move(self, idx, vecinos):
        """Aplica la política propia del fantasma `idx` para elegir su siguiente celda."""
        policy = self.ghost_policies[idx]

        # 1) Política personalizada mediante función callable
        if callable(policy):
            chosen = policy(self, idx, vecinos)
            if chosen in vecinos:
                return chosen

        # 2) Política numérica: probabilidad individual de persecución directa
        if isinstance(policy, (float, int)):
            if self.np_random.random() < float(policy):
                candidatos = self._step_towards(vecinos, self.pac)
            else:
                candidatos = vecinos
            return candidatos[self.np_random.integers(len(candidatos))]

        # 3) Políticas por nombre
        if policy == "chaser":
            # Persigue directamente a Pac-Man con probabilidad `ghost_chase_prob`
            if self.np_random.random() < self.ghost_chase_prob:
                candidatos = self._step_towards(vecinos, self.pac)
            else:
                candidatos = vecinos

        elif policy == "ambusher":
            # Apunta 3 pasos adelante de la dirección actual de Pac-Man
            if self.np_random.random() < self.ghost_chase_prob:
                dr, dc = self.pac_dir
                target = self._nearest_free_cell(
                    self.pac[0] + 3 * dr, self.pac[1] + 3 * dc
                )
                candidatos = self._step_towards(vecinos, target)
            else:
                candidatos = vecinos

        elif policy == "patrol":
            # Si está a más de 4 celdas persigue a Pac-Man; si está cerca se repliega a una esquina
            dist_to_pac = self._bfs(self.pac)[self.ghosts[idx]]
            if dist_to_pac > 4 and self.np_random.random() < self.ghost_chase_prob:
                candidatos = self._step_towards(vecinos, self.pac)
            else:
                corner_r = 1 if self.pac[0] > self.H // 2 else self.H - 2
                corner_c = 1 if self.pac[1] > self.W // 2 else self.W - 2
                target = self._nearest_free_cell(corner_r, corner_c)
                candidatos = self._step_towards(vecinos, target)

        elif policy == "random":
            # Movimiento aleatorio evitando retroceder inmediatamente si hay otro camino
            prev_pos = self.prev_ghosts[idx]
            no_backtrack = [v for v in vecinos if v != prev_pos]
            candidatos = no_backtrack if no_backtrack else vecinos

        else:
            candidatos = vecinos

        return candidatos[self.np_random.integers(len(candidatos))]

    def _move_ghosts(self):
        for idx, (r, c) in enumerate(self.ghosts):
            vecinos = [
                (r + dr, c + dc) for dr, dc in self.MOVES if self._free(r + dr, c + dc)
            ]
            if not vecinos:
                continue
            self.prev_ghosts[idx] = (r, c)
            self.ghosts[idx] = self._select_ghost_move(idx, vecinos)

    # ---------- API Gymnasium ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pac = self.pac_start
        self.pac_dir = (0, 1)
        self.ghosts = list(self.ghost_starts)
        self.prev_ghosts = list(self.ghost_starts)
        self.pellets = self.pellets_init.copy()
        self.t = 0
        return self._obs(), {
            "pellets_left": int(self.pellets.sum()),
            "num_ghosts": len(self.ghosts),
            "ghost_policies": list(self.ghost_policies),
        }

    def step(self, action):
        self.t += 1
        reward = self.r_step
        terminated = truncated = False
        win = caught = False

        use_shaping = (
            self.r_closer_pellet != 0.0 or self.r_farther_ghost != 0.0
        )
        if use_shaping:
            prev_pellet_dist, prev_ghost_dist = self._min_pellet_and_ghost_dist(
                self.pac, self.pellets, self.ghosts
            )

        # 1) movimiento de Pac-Man
        dr, dc = self.MOVES[int(action)]
        self.pac_dir = (dr, dc)
        old_pac = self.pac
        old_ghosts = list(self.ghosts)
        nr, nc = old_pac[0] + dr, old_pac[1] + dc
        if self._free(nr, nc):
            self.pac = (nr, nc)
        else:
            reward += self.r_wall

        # 2) comer bolita
        ate_pellet = False
        if self.pellets[self.pac]:
            self.pellets[self.pac] = False
            ate_pellet = True
            eaten_ratio = 1.0 - (float(self.pellets.sum()) / float(self.total_pellets_init))
            reward += self.r_pellet + self.r_progress_coef * eaten_ratio

        if self.pellets.sum() == 0:  # ganó (tiene prioridad sobre los fantasmas)
            reward_remaining = self.max_steps - self.t
            reward += self.r_win + reward_remaining * self.r_win_remaining_coef
            terminated = win = True
        elif self.pac in self.ghosts:  # Pac-Man caminó hacia algún fantasma
            caught = True
        else:
            # 3) movimiento de cada fantasma según su política individual
            self._move_ghosts()
            for g_new, g_old in zip(self.ghosts, old_ghosts):
                if self.pac == g_new or (self.pac == g_old and g_new == old_pac):
                    caught = True  # incluye cruzarse "de frente" con cualquier fantasma
                    break

        # 4) Reward shaping informativo opcional (acercarse a bolitas / alejarse de fantasmas)
        if use_shaping and not terminated and not caught:
            new_pellet_dist, new_ghost_dist = self._min_pellet_and_ghost_dist(
                self.pac, self.pellets, self.ghosts
            )
            if self.r_closer_pellet != 0.0 and not ate_pellet:
                reward += self.r_closer_pellet * (prev_pellet_dist - new_pellet_dist)
            if self.r_farther_ghost != 0.0 and prev_ghost_dist <= 5.0:
                reward += self.r_farther_ghost * (new_ghost_dist - prev_ghost_dist)

        if caught:
            reward += self.r_dead
            terminated = True

        if not terminated and self.t >= self.max_steps:
            truncated = True

        info = {
            "pellets_left": int(self.pellets.sum()),
            "win": win,
            "caught": caught,
            "num_ghosts": len(self.ghosts),
        }
        return self._obs(), float(reward), terminated, truncated, info

    def _min_pellet_and_ghost_dist(self, pac_pos, pellets_grid, ghosts_list):
        dist_map = self._bfs(pac_pos)
        max_d = float(self.H + self.W)
        pellet_coords = np.argwhere(pellets_grid)
        if len(pellet_coords) > 0:
            min_p = min(float(dist_map[tuple(p)]) for p in pellet_coords)
            if np.isinf(min_p):
                min_p = max_d
        else:
            min_p = 0.0

        min_g = min((float(dist_map[g]) for g in ghosts_list), default=max_d)
        if np.isinf(min_g):
            min_g = max_d
        return min_p, min_g

    # ---------- render ----------
    def render(self):
        if self.render_mode == "ansi":
            filas = []
            ghost_set = set(self.ghosts)
            for r in range(self.H):
                fila = ""
                for c in range(self.W):
                    if (r, c) == self.pac:
                        fila += "P"
                    elif (r, c) in ghost_set:
                        fila += "G"
                    elif self.walls[r, c]:
                        fila += "#"
                    elif self.pellets[r, c]:
                        fila += "."
                    else:
                        fila += " "
                filas.append(fila)
            return "\n".join(filas)
        return self._draw_rgb()

    def _draw_rgb(self, S=32):
        img = np.zeros((self.H * S, self.W * S, 3), np.uint8)
        yy, xx = np.mgrid[0:S, 0:S]
        circ = lambda rad: (yy - S / 2 + 0.5) ** 2 + (xx - S / 2 + 0.5) ** 2 <= rad**2
        big, small = circ(S * 0.38), circ(S * 0.12)
        for r in range(self.H):
            for c in range(self.W):
                tile = img[r * S : (r + 1) * S, c * S : (c + 1) * S]
                if self.walls[r, c]:
                    tile[:] = (33, 33, 222)
                elif self.pellets[r, c]:
                    tile[small] = (255, 255, 255)
                if (r, c) == self.pac:
                    tile[big] = (255, 221, 0)

        # Dibujar cada fantasma con un color distintivo según su índice/política
        for idx, (gr, gc) in enumerate(self.ghosts):
            tile = img[gr * S : (gr + 1) * S, gc * S : (gc + 1) * S]
            color = self.GHOST_COLORS[idx % len(self.GHOST_COLORS)]
            tile[big] = color
        return img


if __name__ == "__main__":
    print("=== Prueba 1: Mapa clásico con 1 fantasma ===")
    env = PacmanEnv(MAPA, render_mode="ansi")
    check_env(env, warn=True)
    obs, info = env.reset(seed=0)
    print(f"Fantasmas: {info['num_ghosts']} | Políticas: {info['ghost_policies']}")

    print("\n=== Prueba 2: Mapa con múltiples fantasmas y políticas distintas ===")
    multi_ghost_env = PacmanEnv(
        MAPA_MULTI_GHOST,
        ghost_policies=["chaser", "ambusher", "patrol"],
        render_mode="ansi",
    )
    check_env(multi_ghost_env, warn=True)

    obs, info = multi_ghost_env.reset(seed=0)
    print(
        f"Fantasmas: {info['num_ghosts']} | Políticas: {info['ghost_policies']}\n"
    )
    print(multi_ghost_env.render(), "\n")
    total, done = 0, False
    while not done:
        obs, r, term, trunc, info = multi_ghost_env.step(
            multi_ghost_env.action_space.sample()
        )
        total += r
        done = term or trunc
    print(multi_ghost_env.render())
    print(f"\nPasos: {multi_ghost_env.t} | recompensa: {total:.2f} | info: {info}")
