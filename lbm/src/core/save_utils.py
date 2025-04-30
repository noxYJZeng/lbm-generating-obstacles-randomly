import os
import numpy as np
import matplotlib.pyplot as plt

from lbm.src.plot.plot import plot_norm, plot_contour

def save_simulation(lattice, obstacles, output_it, base_output_dir, dx, dy, dpi=100, save_contour=True):
    os.makedirs(base_output_dir, exist_ok=True)

    npz_dir = os.path.join(base_output_dir, "npz")
    os.makedirs(npz_dir, exist_ok=True)

    norm_img_dir = os.path.join(base_output_dir, "images_norm")
    os.makedirs(norm_img_dir, exist_ok=True)

    lattice.norm_img_dir = norm_img_dir

    obstacle_map = lattice.lattice.copy()
    nx, ny = lattice.nx, lattice.ny

    for obs in obstacles:
        type_id = {
            "cylinder": 1,
            "square":   2,
            "prism1":   3,
            "prism2":   4,
            "ellipse":  5,
            "star":     6,
            "heart":    7,
            "hexagon":  8
        }[obs.type]

        i = int((obs.pos[0] - lattice.x_min) / dx)
        j = int((obs.pos[1] - lattice.y_min) / dy)
        if 0 <= i < nx and 0 <= j < ny:
            obstacle_map[i, j] = type_id

    npz_filename = os.path.join(npz_dir, f"output_data_{output_it:04d}.npz")
    np.savez_compressed(
        npz_filename,
        velocity=lattice.u,
        density=lattice.rho,
        lattice_map=obstacle_map
    )
    print(f"[Saved .npz] {npz_filename}")

    plot_norm(lattice, val_min=0.0, val_max=1.5, output_it=output_it, dpi=dpi)
    print(f"[Saved norm image] Iter {output_it}")

    # if save_contour:
    #     lattice.contour_img_dir = contour_img_dir
    #     plot_contour(lattice, output_it=output_it, dpi=dpi)
    #     print(f"[Saved contour image] Iter {output_it}")

    lattice.generate_image(obstacles)
    print(f"[Saved lattice image] Iter {output_it}")

    images_folder = os.path.join(base_output_dir, "images")
    if os.path.exists(images_folder) and not os.listdir(images_folder):
        os.rmdir(images_folder)
        print(f"[Deleted empty folder] {images_folder}")