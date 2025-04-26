import math
import random
import numpy as np
import os

from lbm.src.app.base_app  import *
from lbm.src.core.lattice  import *
from lbm.src.core.obstacle import *
from lbm.src.utils.buff    import *
from lbm.src.plot.plot     import *

###############################################
# A class that creates 10 random obstacles without overlap
###############################################
class random10(base_app):
    def __init__(self):
        self.name    = "random10"
        self.Re_lbm  = 500.0
        self.L_lbm   = 1024
        self.u_lbm   = 0.02
        self.rho_lbm = 1.0
        self.t_max   = 10.0
        self.x_min   = -1.0
        self.x_max   = 10.24
        self.y_min   = -1.0
        self.y_max   = 10.24

        # IBB + other LBM options
        self.IBB         = True
        self.stop        = 'it'
        self.obs_cv_ct   = 1.0e-3
        self.obs_cv_nb   = 1000
        self.n_obs       = 10    # 10 obstacles
        self.output_freq = 500
        self.output_it   = 0
        self.dpi         = 200

        self.compute_lbm_parameters()

        # List of placed obstacles
        self.obstacles = []
        # Randomly generate them
        self.add_random_obstacles()
        
    def compute_lbm_parameters(self):

        self.Cs      = 1.0 / math.sqrt(3.0)
        self.ny      = self.L_lbm
        self.u_avg   = 2.0 * self.u_lbm / 3.0
        self.nu_lbm  = self.u_avg * self.L_lbm / self.Re_lbm
        self.tau_lbm = 0.5 + self.nu_lbm / (self.Cs**2)
        self.dt      = self.Re_lbm * self.nu_lbm / self.L_lbm**2
        self.dx      = (self.y_max - self.y_min) / self.ny
        self.dy      = self.dx
        self.nx      = math.floor(
            self.ny * (self.x_max - self.x_min) / (self.y_max - self.y_min)
        )
        self.it_max  = math.floor(self.t_max / self.dt)
        self.sigma   = math.floor(10 * self.nx)

    def bounding_circle_radius(self, shape_type, size):

        # Approximate bounding circle radius for each shape, depending on how 'size' is interpreted in code.

        if shape_type == "cylinder":
            # size is the cylinder radius
            return size
        elif shape_type == "square":
            # size is the side length
            # bounding circle radius = half the diagonal = side * sqrt(2)/2
            return size * math.sqrt(2) / 2
        elif shape_type in ["prism1", "prism2"]:
            return size / math.sqrt(3)
        else:
            return size

    def is_overlapping(self, x1, y1, r1, x2, y2, r2):
        """
        Check if two circles overlap: circle1 center (x1,y1), radius r1,
        circle2 center (x2,y2), radius r2.
        Overlap if distance < (r1 + r2).
        """
        dx = x1 - x2
        dy = y1 - y2
        dist_sq = dx*dx + dy*dy
        r_sum   = r1 + r2
        # Compare squared distance to squared radii sum
        return dist_sq < (r_sum*r_sum)

    def add_random_obstacles(self):
        """
        Randomly pick shapes from a list of possible obstacle types
        and place them in random positions within the domain, avoiding overlap.
        """
        possible_shapes = ["cylinder", "square", "prism1", "prism2"]

        max_attempts_per_obs = 100  # avoid infinite loops

        for i in range(self.n_obs):
            placed = False
            attempts = 0

            while not placed and attempts < max_attempts_per_obs:
                shape_type = random.choice(possible_shapes)

                # Number of corner points might differ by shape
                if shape_type == "cylinder":
                    n_pts = 200
                elif shape_type == "square":
                    n_pts = 4
                elif shape_type in ("prism1", "prism2"):
                    n_pts = 3
                else:
                    n_pts = 2  # fallback

                # Random size
                size = random.uniform(0.1, 2.0)

                # bounding circle radius
                r_bound = self.bounding_circle_radius(shape_type, size)

                # Random center position within domain, accounting for bounding circle
                x_pos = random.uniform(self.x_min + r_bound, self.x_max - r_bound)
                y_pos = random.uniform(self.y_min + r_bound, self.y_max - r_bound)

                # Check overlap with previously placed obstacles
                overlap_found = False
                for existing_obs in self.obstacles:
                    # bounding circle for existing
                    existing_r_bound = self.bounding_circle_radius(
                        existing_obs.type, existing_obs.size
                    )
                    ex_x_pos, ex_y_pos = existing_obs.pos

                    # if overlap, break
                    if self.is_overlapping(
                        x_pos, y_pos, r_bound,
                        ex_x_pos, ex_y_pos, existing_r_bound
                    ):
                        overlap_found = True
                        break

                # If no overlap, place this new obstacle
                if not overlap_found:
                    obs = obstacle(
                        name   = f"random_{shape_type}_{i}",
                        n_pts  = n_pts,
                        n_spts = 50,
                        type   = shape_type,
                        size   = size,
                        pos    = [x_pos, y_pos]
                    )
                    self.obstacles.append(obs)
                    placed = True
                attempts += 1

            if not placed:
                print(f"WARNING: Could not place obstacle #{i} after {max_attempts_per_obs} attempts.")

    def initialize(self, lattice):

        self.add_obstacles(lattice, self.obstacles)

        self.set_inlets(lattice, 0)
        lattice.u[:, np.where(lattice.lattice > 0.0)] = 0.0
        lattice.rho *= self.rho_lbm

        lattice.generate_image(self.obstacles)

        lattice.equilibrium()
        lattice.g = lattice.g_eq.copy()

    def set_inlets(self, lattice, it):

        val = it
        ret = (1.0 - math.exp(-val**2/(2.0*self.sigma**2)))

        for j in range(self.ny):
            pt = lattice.get_coords(0, j)
            lattice.u_left[:, j] = ret * self.u_lbm * self.poiseuille(pt)

        # Top/ bottom
        lattice.u_top[0, :]   = 0.0
        lattice.u_bot[0, :]   = 0.0
        # Right boundary
        lattice.u_right[1, :] = 0.0
        lattice.rho_right[:]  = self.rho_lbm

    def set_bc(self, lattice):

        for obs in self.obstacles:
            lattice.bounce_back_obstacle(obs)

        # Common LBM wall/ corner boundary conditions
        lattice.zou_he_bottom_wall_velocity()
        lattice.zou_he_left_wall_velocity()
        lattice.zou_he_top_wall_velocity()
        lattice.zou_he_right_wall_pressure()
        lattice.zou_he_bottom_left_corner()
        lattice.zou_he_top_left_corner()
        lattice.zou_he_top_right_corner()
        lattice.zou_he_bottom_right_corner()

    def outputs(self, lattice, it):
            if (it % self.output_freq != 0):
                return
            print(f"[Output] Iteration {it}")
            print(f"Lattice velocity norm = {np.linalg.norm(lattice.u)} at iteration {it}")



    def poiseuille(self, pt):

        x = pt[0]
        y = pt[1]
        H = self.y_max - self.y_min
        u = np.zeros(2)
        u[0] = 4.0 * (self.y_max - y) * (y - self.y_min) / (H**2)
        return u
