import os
import numpy as np
import matplotlib.pyplot as plt

from lbm.src.plot.plot import plot_norm, plot_contour

def save_simulation(lattice, obstacles, output_it, base_output_dir, dx, dy, dpi=100, save_contour=True):
    # 创建 base_output_dir 目录
    os.makedirs(base_output_dir, exist_ok=True)

    # npz文件目录
    npz_dir = os.path.join(base_output_dir, "npz")
    os.makedirs(npz_dir, exist_ok=True)

    # norm图像目录
    norm_img_dir = os.path.join(base_output_dir, "images_norm")
    os.makedirs(norm_img_dir, exist_ok=True)

    # 将 lattice 需要的 norm_img_dir 记录下来
    lattice.norm_img_dir = norm_img_dir

    # 生成 obstacle map（存入 npz）
    obstacle_map = lattice.lattice.copy()
    nx, ny = lattice.nx, lattice.ny

    for obs in obstacles:
        type_id = {"cylinder": 1, "square": 2, "prism1": 3, "prism2": 4}[obs.type]
        i = int((obs.pos[0] - lattice.x_min) / dx)
        j = int((obs.pos[1] - lattice.y_min) / dy)
        if 0 <= i < nx and 0 <= j < ny:
            obstacle_map[i, j] = type_id

    # 保存 npz文件
    npz_filename = os.path.join(npz_dir, f"output_data_{output_it:04d}.npz")
    np.savez_compressed(
        npz_filename,
        velocity=lattice.u,
        density=lattice.rho,
        lattice_map=obstacle_map
    )
    print(f"[Saved .npz] {npz_filename}")

    # 保存 norm图像
    plot_norm(lattice, val_min=0.0, val_max=1.5, output_it=output_it, dpi=dpi)
    print(f"[Saved norm image] Iter {output_it}")

    # 保留你的 contour 注释 (不启用 contour)
    # if save_contour:
    #     lattice.contour_img_dir = contour_img_dir
    #     plot_contour(lattice, output_it=output_it, dpi=dpi)
    #     print(f"[Saved contour image] Iter {output_it}")

    # 保存 lattice 整体图片（直接保存在 base_output_dir）
    lattice.generate_image(obstacles)
    print(f"[Saved lattice image] Iter {output_it}")

    images_folder = os.path.join(base_output_dir, "images")
    if os.path.exists(images_folder) and not os.listdir(images_folder):
        os.rmdir(images_folder)
        print(f"[Deleted empty folder] {images_folder}")