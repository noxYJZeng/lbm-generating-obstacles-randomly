# lbm – randomly generating obstacles

This project builds upon a basic 2D Lattice-Boltzmann Method (LBM) solver based on D2Q9 lattice and TRT collision.

While the original version supported basic flows like lid-driven cavity and Turek benchmarks, this extended version introduces several new **features** and **enhancements** for flexibility, usability, and application expansion.


## New Features

- **New Obstacles Types**  
  Added prism1, prism2, star, hexagon, heart and ellipse as new obstacle types.

- **Randomized Obstacles Generation**  
  Added `random10` `random3` applications that dynamically generate randomized obstacle configurations on every run. Shapes are randomly selected, placed without overlap, and optionally rotated to create a wide range of simulation setups, enabling generalization testing and diverse dataset creation.

- **Automatic Output Management**  
  All simulation outputs (including `.npz` data and generated images) are automatically saved under a timestamped results folder in `./results/`. The npz files include density, velocity of the flow and lattice map of obstacles. They can be used for Latnet model training.

- **Improved Target Behavior**  
  Fixed target (obstacle) rendering bugs to ensure no unexpected flashes or "teleporting" effects when targets are recycled at domain boundaries.

- **Cooldown-based Target Spawning**  
  Targets now respect a cooldown period after deactivation, preventing immediate reuse and avoiding rendering artifacts.



## Running Simulations

To run a simulation:
```bash
python3 start.py <application_name>
```
Simulation results (including .npz files and images) will be saved automatically under ./results/<timestamp>/.


## Sample Simulation

Below are sample results showing randomly generated obstacles and flow behavior:

<div align="center"> <img src="lbm/save/random10.gif" width="600" alt="Random Obstacles Simulation GIF"> <br><br> <img src="lbm/save/random10_new.gif" width="400" alt="Zoomed Random Simulation GIF"> </div>

## Credits

Original base project by [jviquerat](https://github.com/jviquerat/lbm). 

Extensions and improvements by [noxYJZeng](https://github.com/noxYJZeng/lbm-generating-obstacles-randomly).
