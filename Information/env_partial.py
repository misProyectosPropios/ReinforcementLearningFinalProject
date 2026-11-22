"""
Entorno de Pac-Man con Observabilidad Parcial en Línea Recta (`PartialObsPacmanEnv`).

Cómo funciona la visión en línea recta del agente:
- Desde la celda actual de Pac-Man `(pr, pc)`, el agente mira en las 4 direcciones
  cardinales en línea recta: arriba `(-1, 0)`, abajo `(1, 0)`, izquierda `(0, -1)`
  y derecha `(0, 1)`.
- En cada dirección ve **toda la línea recta del pasillo** hasta chocar contra una
  pared (`#`) (inclusive la pared que frena la línea de visión).
- **No lo tapa el primer elemento**: si en ese pasillo recto hay bolitas (`.`),
  uno o más fantasmas (`G`) y más atrás siguen habiendo bolitas o fantasmas antes
  de la pared, el agente ve **todos** los elementos que estén sobre esa línea recta.
- Todo lo que queda fuera de las 4 líneas rectas desde su posición actual permanece
  oculto (en niebla / `0.0`), a menos que se active `stop_at_walls=False` (modo
  rayos-X donde ve toda la fila `pr` y toda la columna `pc` completas incluso a través
  de paredes internas).

Canales de la observación `(5 * H * W,)`:
  1. `visible_mask`: `1.0` en las celdas actualmente visibles en línea recta, `0.0` en la niebla.
  2. `visible_walls`: paredes visibles dentro de la línea de visión.
  3. `visible_pellets`: todas las bolitas visibles a lo largo de las 4 líneas rectas.
  4. `pac`: posición actual de Pac-Man (`1.0` en `(pr, pc)`).
  5. `visible_ghosts`: todos los fantasmas que estén en cualquiera de las 4 líneas rectas visibles.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

from maps import MAPA, MAPA_MULTI_GHOST
from env import PacmanEnv


class PartialObsPacmanEnv(PacmanEnv):
    """
    Subclase de `PacmanEnv` donde Pac-Man solo observa lo que está en línea recta
    (arriba, abajo, izquierda, derecha) desde su posición actual, viendo todos los
    elementos (bolitas, fantasmas y la pared al final del pasillo) a lo largo de esa línea.
    """

    def __init__(
        self,
        mapa,
        max_steps: int = 200,
        ghost_chase_prob: float = 0.7,
        ghost_policies=None,
        stop_at_walls: bool = True,
        include_visibility_channel: bool = True,
        render_fog: bool = True,
        render_mode=None,
        **reward_kwargs,
    ):
        self.stop_at_walls = stop_at_walls
        self.include_visibility_channel = include_visibility_channel
        self.render_fog = render_fog

        super().__init__(
            mapa,
            max_steps=max_steps,
            ghost_chase_prob=ghost_chase_prob,
            ghost_policies=ghost_policies,
            full_info_obs=False,
            render_mode=render_mode,
            **reward_kwargs,
        )

        self.num_partial_channels = 5 if self.include_visibility_channel else 4
        self.observation_space = spaces.Box(
            0.0,
            1.0,
            shape=(self.num_partial_channels * self.H * self.W,),
            dtype=np.float32,
        )

    def get_visible_mask(self) -> np.ndarray:
        """
        Calcula una máscara booleana `(H, W)` con `True` en todas las celdas que Pac-Man
        puede ver en línea recta en las 4 direcciones desde `self.pac`.

        - Si `stop_at_walls=True` (por defecto): el rayo en cada dirección recorre todas las
          celdas libres en línea recta (atravesando bolitas y fantasmas sin detenerse) y se
          detiene recién cuando encuentra una pared `#` (la pared encontrada también se marca
          como visible para que el agente sepa dónde termina el pasillo).
        - Si `stop_at_walls=False`: ve toda la fila `pr` y toda la columna `pc` de punta a punta.
        """
        mask = np.zeros((self.H, self.W), dtype=bool)
        pr, pc = self.pac
        mask[pr, pc] = True

        if not self.stop_at_walls:
            mask[pr, :] = True
            mask[:, pc] = True
            return mask

        for dr, dc in self.MOVES:
            r, c = pr + dr, pc + dc
            while 0 <= r < self.H and 0 <= c < self.W:
                mask[r, c] = True
                if self.walls[r, c]:
                    # Ve la pared que cierra el pasillo recto, pero no detrás de esa pared
                    break
                # Si hay bolita o fantasma NO corta la visión: sigue viendo toda la línea recta
                r += dr
                c += dc

        return mask

    def _obs(self) -> np.ndarray:
        vis_mask = self.get_visible_mask()
        vis_f = vis_mask.astype(np.float32)

        # Solo se informa el contenido de las celdas que están dentro de la línea recta de visión
        visible_walls = self.walls.astype(np.float32) * vis_f
        visible_pellets = self.pellets.astype(np.float32) * vis_f

        pac = np.zeros((self.H, self.W), dtype=np.float32)
        pac[self.pac] = 1.0

        visible_ghosts = np.zeros((self.H, self.W), dtype=np.float32)
        for g_pos in self.ghosts:
            if vis_mask[g_pos]:
                visible_ghosts[g_pos] = 1.0

        channels = [
            visible_walls.ravel(),
            visible_pellets.ravel(),
            pac.ravel(),
            visible_ghosts.ravel(),
        ]
        if self.include_visibility_channel:
            channels.insert(0, vis_f.ravel())

        return np.concatenate(channels)

    def render(self):
        """
        Renderiza el mapa mostrando opcionalmente con `?` (en modo ANSI) o oscureciendo
        (en modo `rgb_array`) las celdas que están fuera de las líneas rectas de visión.
        """
        vis_mask = self.get_visible_mask()

        if self.render_mode == "ansi":
            filas = []
            ghost_set = set(self.ghosts)
            for r in range(self.H):
                fila = ""
                for c in range(self.W):
                    if not self.render_fog or vis_mask[r, c]:
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
                    else:
                        fila += "?"
                filas.append(fila)
            return "\n".join(filas)

        img = self._draw_rgb()
        if not self.render_fog:
            return img

        # En modo RGB oscurecemos las celdas fuera de la línea de visión de Pac-Man
        S = 32
        fogged = img.copy()
        for r in range(self.H):
            for c in range(self.W):
                if not vis_mask[r, c]:
                    tile = fogged[r * S : (r + 1) * S, c * S : (c + 1) * S]
                    # Ocultamos bolitas/fantasmas fuera de visión y dejamos niebla gris oscura
                    tile[:] = (25, 25, 35)
        return fogged


if __name__ == "__main__":
    print("=== Prueba del Entorno con Observabilidad Parcial en Línea Recta ===")
    env = PartialObsPacmanEnv(MAPA_MULTI_GHOST, render_mode="ansi")
    check_env(env, warn=True)

    obs, info = env.reset(seed=0)
    print(f"Dimensión de observación parcial: {env.observation_space.shape}")
    print("Leyenda ANSI: '?' = fuera de la línea de visión | 'P' = Pac-Man | 'G' = Fantasma visible | '.' = Bolita visible\n")
    print("Estado inicial (vista parcial de Pac-Man):")
    print(env.render(), "\n")

    total_reward = 0.0
    for step_idx in range(1, 6):
        action = env.action_space.sample()
        move_name = ["Arriba", "Abajo", "Izquierda", "Derecha"][int(action)]
        obs, r, term, trunc, info = env.step(action)
        total_reward += r
        print(f"--- Paso {step_idx} (Acción: {move_name} | Recompensa acum: {total_reward:.2f}) ---")
        print(env.render(), "\n")
        if term or trunc:
            break
