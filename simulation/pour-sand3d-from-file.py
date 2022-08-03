import argparse
import os
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams

matplotlib.use('Agg')
rcParams['figure.subplot.left'] = 0.15
rcParams['figure.subplot.right'] = 0.95
rcParams['figure.subplot.bottom'] = 0.12
rcParams['figure.subplot.top'] = 0.95
# rcParams['figure.figsize'] = 6, 6
rcParams['figure.dpi'] = 100
#rcParams['ps.fonttype'] = 12
rcParams['font.size'] = 12

import time
import numpy as np
import taichi as tc


def perform_sim(cup_codimensional, cup_friction, beta_sand, sand_friction_angle,
                rigid_density, rigid_friction, cup_density, cup_angular_damping, sand_density, sand_cohesion, trajectory_file=None):

    # Path to numpy array of shape (301,4) -> [x, y, z, x_rot]
    # If None, then use random movements
    trajectory_file = trajectory_file
    sim_idx = 1
    seed = 2021
    timesteps = 400

    # Take position and rotation from predefined trajectory
    # Allows taking in saved cup trajectories

    def position_function(t):
        i = int(t * 100)
        if i < 100:
            return tc.Vector(trajectory[0,0], trajectory[0,1], trajectory[0,2])
        if i > 399:
            return tc.Vector(trajectory[-1,0], trajectory[-1,1], trajectory[-1,2])
        i = i - 100
        return tc.Vector(trajectory[i,0], trajectory[i,1], trajectory[i,2])

    def rotation_fucntion(t):
        angle_multiplier = 1.0
        i = int(t * 100)
        if i < 100:
            rot_x = angle_multiplier * (rotation[0] - 180)
            return tc.Vector(rot_x, 0, 0)
        if i > 399:
            rot_x = angle_multiplier * (rotation[-1] -180)
            return tc.Vector(rot_x, 0, 0)
        i = i - 100
        rot_x = angle_multiplier * (rotation[i] - 180)
        return tc.Vector(rot_x, 0, 0)

    trajectory = None
    rotation = None

    res = (128, 128, 128)
    mpm = tc.dynamics.MPM(
        res=res,
        penalty=1e4,
        pushing_force=0,
        base_delta_t=1e-4,
        num_frames=timesteps,
        particle_collision=True
        )
    np.random.seed(seed)
    # Randomize trajectory or load from file
    if trajectory_file is None:
        trajectory = np.zeros((301, 3))
        trajectory[0] = np.array([0.5, 0.4, 0.5])
        rotation = np.ones(301) * 180
        # Randomize movement
        for j in range(sim_idx):
            max_rotation = np.random.uniform(90, 140)
            direction = np.random.choice([-1,1])
            speed = np.random.uniform(1,2.5)
            i = 1
            while i < 301:
                trajectory[i] = np.array([0.5, 0.4, trajectory[i - 1, 2] + np.random.normal(scale=0.001)])
                rotation[i] = 180 + direction * np.sin(speed*i*0.01) * max_rotation + np.random.normal(scale=0.2)
                i = i + 1
    else:
        arr = np.load(trajectory_file)
        print("Array shape", arr.shape)
        trajectory = np.zeros((301, 3))
        rotation = np.ones(301) * 180
        trajectory[:, :] = np.array([0.5, 0.4, 0.5])
        trajectory[1:, 2] += arr[:,1]
        rotation[1:] = -np.rad2deg(arr[:,0])
        print(trajectory)
        print(rotation)
    # Define initial position of cup to be the first point in trajectory
    init_pos = trajectory[0]

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
    # Note: Add container first to filter out container particles in postprocessing
    mpm.add_particles(
        type='rigid',
        codimensional=rigid_codimensional,
        density=rigid_density,
        scripted_position=tc.constant_function((0.5,0.189,0.5)),
        scripted_rotation=tc.constant_function((0,0,0)),
        friction=rigid_friction,
        mesh_fn='$mpm/container.obj')

    # Add sand to the scene
    # Translation shifts the object from point (0.5,0.5)
    density_factor = 4.0        # The higher the value the more particles
    tex = tc.Texture('rect', bounds=(0.045, 0.1, 0.045)) * density_factor
    tex = tex.translate((init_pos[0] - 0.5, init_pos[1] - 0.45, init_pos[2] - 0.5))

    mpm.add_particles(
        type='sand',
        pd=True,
        friction_angle=sand_friction_angle,
        cohesion=sand_cohesion,
        density_tex=tex.id,
        initial_velocity=(0,0,0),
        density=sand_density,
        color=(0.9, 0.5, 0.6),
        beta=beta_sand
    )

    mesh_fn = '$mpm/wine_glass_new.obj'
    # Add cup mesh
    # Note: Add cup mesh after sand not to filter out cup particles
    mpm.add_particles(
        type='rigid',
        codimensional=cup_codimensional,  # Thin shell or not
        density=cup_density,  # 40 for thin shell, 400 for non-thin shell
        scripted_position=tc.function13(position_function),
        scripted_rotation=tc.function13(rotation_fucntion),
        angular_damping=cup_angular_damping,  # damping of angular velocity, typical value 1
        friction=cup_friction,
        scale=(1.0,1.0,1.0),
        mesh_fn=mesh_fn)
        # linear damping default 0, typical value 1
        # restitution coefficient for restitution, default
        # friction coefficient. sticky: -1, slip: -2,
        # slip with friction -2.4, means coeff of friction 0.4 with slip
    # Simulate
    start_time = time.time()
    mpm.simulate(
        clear_output_directory=True,
        update_frequency=1,
        print_profile_info=False
    )
    sim_time = time.time() - start_time
    print(f'MPM simulation takes {sim_time:.2f} seconds')
    #print(f'Direction: {direction}; Maximum angle: {max_rotation}')
    datadir = os.environ['TAICHI_REPO_DIR'] +'/outputs/mpm/pour-sand3d-from-file/frames/'
    material_id = 1
    main_folder = os.environ['TAICHI_REPO_DIR'] + '/outputs/mpm/'
    idx_folder=0
    # Create the target directory
    while True:
        target_dir = main_folder + str(idx_folder)
        target_exists = os.path.isdir(target_dir)
        if target_exists:
            print("Directory exists")
            idx_folder+=1
        else:
            print("Creating new directory: " + target_dir)
            os.makedirs(target_dir)
            break

    def plot_multiple_nodes(sand_particles, rigid_particles, title, desired=None):
        # Plot YZ
        fig, ax4 = plt.subplots()
        ax4.scatter(sand_particles[:, 2], sand_particles[:, 1], color='darkorange', zorder=4, label='Sand Particles')
        ax4.scatter(rigid_particles[:, 2], rigid_particles[:, 1], color='grey', zorder=3, label='Rigid Particles')
        if desired is not None:
            ax4.scatter(desired[:, 2], desired[:, 1], color='darkkhaki', zorder=2, label='Desired')
        ax4.set_xlabel("Y")
        ax4.set_ylabel("Z")
        ax4.set_xlim(0.0, 1.0)
        ax4.set_ylim(0.0, 0.8)
        ax4.legend(frameon=True)
        plt.axvline(x=0.4)
        plt.axvline(x=0.6)
        plt.axhline(y=0.1)
        plt.savefig(title+'particles_YZ.png', dpi=600)
        plt.close(fig)

    def generate_plots(data_dir, timesteps, material_id, target_dir):
        # Find files in dataset directory
        file_names = [file for file in os.listdir(data_dir) if '.csv' in file]
        sim_data = np.zeros((len(file_names), 2))
        for idx, file_name in enumerate(file_names):
            # Load data and reshape it
            df_data = pd.read_csv(os.path.join(data_dir, file_name), header=None)
            data_dim = df_data.shape[1]
            data = np.array(df_data).reshape(timesteps, -1, data_dim)
            nof_particles = data.shape[1]
            # Filter container particles out of data
            # (The container particles are rigid body particles that are added to the scene before sand particles)
            first_sand_id = None
            for i in range(nof_particles):
                if data[0, i, material_id] < 0.5:
                    first_sand_id = i
                    break
        # Plot the data
        data = data[:,first_sand_id:,:]
        material_id = 1
        cartesian_idx = [2, 3, 4]
        rigid_particles_idx = data[0, :, material_id] == 1
        sand_particles_idx = data[0, : ,material_id] == 0
        plot_indexes = [255, 399]
        for i in range(100, 400):  # for all the timesteps
            rigid_i = data[i, rigid_particles_idx]
            sand_i = data[i, sand_particles_idx]
            str_i = str(i)
            # if i < 10:
            #     str_i = "00"+str(i)
            # elif i < 100:
            #     str_i = "0"+str(i)
            plot_multiple_nodes(sand_i[:, cartesian_idx],
                                rigid_i[:, cartesian_idx], title=target_dir+'/Simulation_'+str_i+"_")

        return

    #Generate the plots
    generate_plots(datadir, timesteps, material_id, target_dir)

    with open(target_dir+'/params.txt', 'w') as f:
        text = " cup_codimensional=[" + str(cup_codimensional) + "]\n cup friction= " + str(cup_friction) + \
            "]\n beta sand=[" + str(beta_sand) + "]\n sand friction angle=[" + str(sand_friction_angle) + \
            "]\n" + str(sand_density) + "]"
        f.write(text)

    import subprocess
    bashCmd = ['cp', '-r', datadir, target_dir]
    # bashCmd = ['ffmpeg', '-i', target_dir+"/Simulation_%03d_particles_YZ.png", '-s', '600x600', '-vcodec',
    #            'libx264', '-crf', '25', target_dir+"/sim.mp4"]
    subprocess.run(bashCmd)
    subprocess.run(['cp', '-r', datadir + '/positions.csv', target_dir])
    # output, error = process.communicate()


# Cup parameters
cup_angular_damping = 2.7

# Sand parameters
sand_cohesion = 0

# Arguments
beta_sand = 1.0
sand_friction_angle = 20
cup_codimensional = True
cup_friction = -2.2

parser = argparse.ArgumentParser(description='Generates rollout prediction for given model.')
parser.add_argument('--angle', default=sand_friction_angle, type=int)
parser.add_argument('--beta', default=beta_sand, type=float)
parser.add_argument('--friction', default=-2.2, type=float)
parser.add_argument('--sand_density', default=400, type=int)
parser.add_argument('--rigid_density', default=40, type=int)
parser.add_argument('--codim', action='store_true')
# parser.add_argument('--trajectory_file', default=None)
parser.add_argument('--trajectory_file',
                    default="/home/mulerod1/projects/sand_simulation/30Dec_tests_results/trajectory_feasible_sim_1.npy")
args = parser.parse_args()
# args.trajectory_file = "/home/mulerod1/projects/sand_simulation/30Dec_tests_results/trajectory_feasible_sim_1.npy"

cup_codimensional = rigid_codimensional = args.codim
cup_friction = rigid_friction = float(args.friction)
beta_sand = float(args.beta)
sand_friction_angle = int(args.angle)
sand_density = int(args.sand_density)
rigid_density = cup_density = args.rigid_density
trajectory_file = args.trajectory_file
print("Codim=", cup_codimensional, ". Cup friction=", cup_friction,
      ". Beta sand=", beta_sand, ". Angle=", sand_friction_angle)


# perform_sim(cup_codimensional, cup_friction, beta_sand, sand_friction_angle,
#             rigid_density, rigid_friction, cup_density, cup_angular_damping, sand_density, sand_cohesion, trajectory_file=trajectory_file)

folder = '/home/mulerod1/projects/mpm-simulations/assets/trajectories/'
i = 18
idx = ('0' + str(i)) if i < 10 else str(i)
# traj_name = folder + 'test_trajectory' + idx + '.npy'
traj_name = args.trajectory_file
perform_sim(cup_codimensional, cup_friction, beta_sand, sand_friction_angle, rigid_density, rigid_friction,
            cup_density, cup_angular_damping, sand_density, sand_cohesion, trajectory_file=traj_name)

