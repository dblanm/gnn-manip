# MPM Simulations

This repository contains several files to simulate sand particles using taichi-mpm. To use these you need to have installed taichi legacy and taichi-mpm.

## 3D Simulations

To run `sand-pour3d.py` you need to copy object meshes from `assets/meshes` to your own `$TAICHI_DIR/assets/mpm`.

## Post processing:

```shell
python postprocess_csv.py -d --target_dir --timesteps 400 --cartesian_idx 2 3 4
```
