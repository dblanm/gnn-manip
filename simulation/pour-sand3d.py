import time
import numpy as np
import taichi as tc
import argparse

# Simulation
#   1. Cup is still
#   2. Sand is poured to center
#   3. Sand is poured to left
#   4. Sand is poured to right
#   5. Part of sand is poured to left and rest to right

parser = argparse.ArgumentParser(description='Generates rollout prediction for given model.')
parser.add_argument('--angle', type=int)
parser.add_argument('--beta', type=float)
parser.add_argument('--friction', default=-2.2, type=float)
parser.add_argument('--sand_density', default=400, type=int)
parser.add_argument('--rigid_density', default=40, type=int)
parser.add_argument('--codim', action='store_true')
parser.add_argument('--sim_idx', default=1, type=int)
parser.add_argument('--density_factor', default=1.0, type=float)
args = parser.parse_args()

angle = args.angle
beta = args.beta
friction = args.friction
sand_density = args.sand_density
rigid_density = args.rigid_density
codim = args.codim
simulation_id = args.sim_idx
density_factor = args.density_factor

use_rectangular_cup = False
use_old_glass = False

init_pos = (0.5, 0.4, 0.5)
init_rot = 0

end_pos = (0.5, 0.4, 0.5)
end_rot = 0

if simulation_id == 2:
    # Pour all sand to center
    end_pos = (0.5, 0.4, 0.625)
    end_rot  = -135
elif simulation_id == 3:
    # Pour all sand to left
    end_pos = (0.5, 0.4, 0.55)
    end_rot  = -135
elif simulation_id == 4:
    # Pour all sand to right
    end_pos = (0.5, 0.4, 0.5)
    end_rot  = 135
elif simulation_id == 5:
    # Pour part of the sand left and rest to right
    end_pos = (0.5, 0.4, 0.5)
    end_rot  = -115
elif simulation_id == 6:
    end_pos = (0.5, 0.4, 0.55)
    end_rot  = -135

def position_function(t):
    if t < 1:
        return tc.Vector(init_pos)
    t = t - 1
    diff_z = end_pos[2] - init_pos[2]
    if simulation_id < 6:
        if t < 1:
            return tc.Vector(init_pos[0], init_pos[1], init_pos[2] + diff_z * (t - 0.01))
        elif t < 2:
            return tc.Vector(init_pos[0], init_pos[1], init_pos[2] + diff_z)
        else:
            return tc.Vector(init_pos[0], init_pos[1], init_pos[2] + diff_z * (3.01 - t))
    else:
        if t < 0.5:
            return tc.Vector(init_pos[0], init_pos[1], init_pos[2] + diff_z * 2 * (t - 0.01))
        elif t < 1.0:
            return tc.Vector(init_pos[0], init_pos[1], init_pos[2] + diff_z)
        elif t < 2.0:
            return tc.Vector(init_pos[0], init_pos[1], init_pos[2] + diff_z - diff_z * 2 * (t - 1.01))
        elif t < 2.5:
            return tc.Vector(init_pos[0], init_pos[1], init_pos[2] - diff_z)
        else:
            return tc.Vector(init_pos[0], init_pos[1], init_pos[2] - diff_z + diff_z * 2 * (t - 2.51))

def rotation_fucntion(t):
    if t < 1:
        return tc.Vector(init_rot, 0, 0)
    t = t - 1
    diff_rot = end_rot - init_rot
    if simulation_id < 5:
        if t < 1:
            return tc.Vector(init_rot, 0, 0)
        elif t < 2:
            return tc.Vector(init_rot + diff_rot * (t-1), 0, 0)
        else:
            return tc.Vector(init_rot + diff_rot * (3-t), 0, 0)
    elif simulation_id == 5:
        vel_rot = diff_rot / 0.75
        if t < 0.75:
            return tc.Vector(init_rot + vel_rot * t, 0, 0)
        elif t < 2.50:
            return tc.Vector(init_rot + diff_rot - vel_rot * (t - 0.75), 0, 0)
        else:
            return tc.Vector(init_rot + diff_rot - 1.75 * vel_rot + 2 * vel_rot * (t - 2.50), 0, 0)
    else:
        if t < 0.5:
            return tc.Vector(init_rot, 0, 0)
        elif t < 1.0:
            return tc.Vector(init_rot + diff_rot * 2 * (t - 0.5), 0, 0)
        elif t < 1.5:
            return tc.Vector(init_rot + diff_rot - diff_rot * 2 * (t - 1.0), 0, 0)
        elif t < 2.0:
            return tc.Vector(init_rot, 0, 0)
        elif t < 2.5:
            return tc.Vector(init_rot - (diff_rot + 20) * 2 * (t - 2.0), 0, 0)
        else:
            return tc.Vector(init_rot - (diff_rot + 20) + (diff_rot + 10) * 2 * (t - 2.5), 0, 0)
        

use_platform = True
res = (128, 128, 128)

mpm = tc.dynamics.MPM(
    res=res, 
    penalty=1e4, 
    pushing_force=0, 
    base_delta_t=1e-4, 
    num_frames=400,
    particle_collision=True
    )

# Add scene boundaries
levelset = mpm.create_levelset()
levelset.add_plane(tc.Vector(0,1,0), -0.1)
levelset.add_plane(tc.Vector(0,-1, 0), 0.9)
levelset.add_plane(tc.Vector(1,0,0), -0.1)
levelset.add_plane(tc.Vector(-1,0,0), 0.9)
levelset.add_plane(tc.Vector(0,0,1), -0.1)
levelset.add_plane(tc.Vector(0,0,-1), 0.9)
levelset.set_friction(-1)
mpm.set_levelset(levelset, False)

# Add container mesh
mpm.add_particles(
    type='rigid',
    codimensional=True,
    density=rigid_density,
    scripted_position=tc.constant_function((0.5,0.189,0.5)),
    scripted_rotation=tc.constant_function((0,0,0)),
    friction=friction,
    mesh_fn='$mpm/container.obj')

# Add sand to the scene
# Translation shifts the object from point (0.5,0.5)
if use_old_glass:
    tex = tc.Texture('rect', bounds=(0.05, 0.065, 0.05 ))
    tex = tex.translate((init_pos[0] - 0.5, init_pos[1] - 0.475, init_pos[2] - 0.5))
else:
    tex = tc.Texture('rect', bounds=(0.045, 0.1, 0.045 )) * density_factor
    tex = tex.translate((init_pos[0] - 0.5, init_pos[1] - 0.45, init_pos[2] - 0.5))

mpm.add_particles(
    type='sand',
    pd=True,
    friction_angle=angle,
    cohesion=0,
    density_tex=tex.id,
    initial_velocity=(0,0,0),
    density=sand_density,
    color=(0.9, 0.5, 0.6)
)

if use_old_glass:
    mesh_fn = '$mpm/rectangular_cup.obj' if use_rectangular_cup else '$mpm/wine_glass.obj'
    # Add cup mesh
    mpm.add_particles(
        type='rigid',
        codimensional=codim,
        density=rigid_density,
        scripted_position=tc.function13(position_function),
        scripted_rotation=tc.function13(rotation_fucntion),
        angular_damping=2.7,
        friction=friction,
        scale=(0.5,0.5,0.5),
        mesh_fn=mesh_fn)
else:
    mesh_fn = '$mpm/rectangular_cup.obj' if use_rectangular_cup else '$mpm/wine_glass_new.obj'
    # Add cup mesh
    mpm.add_particles(
        type='rigid',
        codimensional=codim,
        density=rigid_density,
        scripted_position=tc.function13(position_function),
        scripted_rotation=tc.function13(rotation_fucntion),
        angular_damping=2.7,
        friction=friction,
        scale=(1.0,1.0,1.0),
        mesh_fn=mesh_fn)

# Simulate
start_time = time.time()
mpm.simulate(
    clear_output_directory=True,
    update_frequency=1,
    print_profile_info=False
)
sim_time = time.time() - start_time
print(f'MPM simulation takes {sim_time:.2f} seconds')