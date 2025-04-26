# lbm – randomly generating obstacles

This project builds upon a basic 2D Lattice-Boltzmann Method (LBM) solver based on D2Q9 lattice and TRT collision.

While the original version supported basic flows like lid-driven cavity and Turek benchmarks, this extended version introduces several new **features** and **enhancements** for flexibility, usability, and application expansion.


## New Features

- **New Obstacles Types**  
  Added prism1 and prism2 as new obstacle types.

- **Randomized Obstacles Generation**  
  Added `random10` `random3`application that dynamically generates random obstacles for each simulation. This enables diverse dataset creation and generalization testing.

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


##  Sample Simulation

Below is a sample result showing randomly generated obstacles and flow behavior:

<p align="center"> <img src="lbm/save/random10.gif" width="600" alt="Random Obstacles Simulation GIF"> </p>


##  Sample Simulation

Original base project by [jviquerat](https://github.com/jviquerat/lbm). 

Extensions and improvements by noxYJZeng.
