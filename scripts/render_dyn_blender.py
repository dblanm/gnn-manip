import bpy
import bmesh

import math
import mathutils

import os
import sys
import csv
import argparse

MATERIAL_COLORS = {
    "Sand":      (0.8, 0.575, 0, 1),
    "Rigid":     (0.8, 0.8, 0.8, 1),
    "Container": (1, 1, 1, 1), # (0.33, 0.33, 0.33, 1)
    "Table":     (0.5, 0.5, 0.5, 1)
}

CAMERA_POSITIONS = [
    ((-0.05, 0.2, 0.6), tuple([i * (3.14/180) for i in (60, 0 ,300)])),
    ((-0.15, 0.5, 0.5), tuple([i * (3.14/180) for i in (75, 0 ,270)])),
    ((0.0, 0.5, 0.2), tuple([i * (3.14/180) for i in (90, 0 ,270)])),
    ((0.5, 0.5, 0.6), tuple([i * (3.14/180) for i in (0, 0 ,270)])),
    ((1.0, 0.5, 0.2), tuple([i * (3.14/180) for i in (90, 0 ,-270)])),
    ((0.5, 0.5, 0.6), tuple([i * (3.14/180) for i in (0, 0 ,180)])),
    ((1.05, 0.8, 0.6), tuple([i * (3.14/180) for i in (60, 0 ,-240)])),
    ((1.15, 0.5, 0.5), tuple([i * (3.14/180) for i in (75, 0 ,-270)])),
]

def auto_int(x):
    return int(x, 0)

def hex_to_rgb(hex_color):
    b = (hex_color & 0xFF) / 255.0
    g = ((hex_color >> 8) & 0xFF) / 255.0
    r = ((hex_color >> 16) & 0xFF) / 255.0
    return r, g, b, 1

def get_materials(colors=MATERIAL_COLORS):
    # Create materials if needed
    mat_sand = bpy.data.materials.get("Sand")
    if mat_sand is None:
        # Create sand material
        mat_sand = bpy.data.materials.new(name="Sand")
    mat_sand.diffuse_color = colors["Sand"]
    mat_rigid = bpy.data.materials.get("Rigid")
    if mat_rigid is None:
        # Create sand material
        mat_rigid = bpy.data.materials.new(name="Rigid")
    mat_rigid.diffuse_color = colors["Rigid"]
    mat_container = bpy.data.materials.get("Container")
    if mat_container is None:
        # Create sand material
        mat_container = bpy.data.materials.new(name="Container")
    mat_container.diffuse_color = colors["Container"]
    mat_table = bpy.data.materials.get("Table")
    if mat_table is None:
        # Create sand material
        mat_table = bpy.data.materials.new(name="Table")
    mat_table.diffuse_color = colors["Table"]
    return mat_sand, mat_rigid, mat_container, mat_table

def set_camera(position, orientation, resolution=(512, 512), transparent_background=False):
    # Set camera position and rotation
    cam_obj = bpy.data.objects["Camera"]
    cam_obj.location = position
    cam_obj.rotation_euler = orientation
    # Set render resolution
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    # Set render engine
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    # Set transparent background
    if transparent_background:
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    
    
def set_background_objects(mat_container, mat_table, args=None):
    if args is not None and args.hide_background_objects:
        return 
    # Create new collection for background objects
    collection = bpy.data.collections.new('Background') 
    bpy.context.scene.collection.children.link(collection)
    # Create mesh for container
    cont_mesh = bpy.data.meshes.new('Container')
    bm = bmesh.new()
    x1 = bm.verts.new((0.45,0.4,0.1))
    x2 = bm.verts.new((0.45,0.6,0.1))
    x3 = bm.verts.new((0.55,0.4,0.1))
    x4 = bm.verts.new((0.55,0.6,0.1))
    x5 = bm.verts.new((0.55,0.4,0.3))
    x6 = bm.verts.new((0.55,0.6,0.3))
    x7 = bm.verts.new((0.45,0.6,0.3))
    x8 = bm.verts.new((0.45,0.4,0.3))
    bmesh.ops.contextual_create(bm, geom=[x1, x2, x3, x4]) # Add bottom
    if args is None or args.camera_idx not in [4,5,6,7]:
        bmesh.ops.contextual_create(bm, geom=[x3, x4, x5, x6]) # Add long side
        bmesh.ops.contextual_create(bm, geom=[x2, x4, x6, x7]) # Add short side
    else:
        bmesh.ops.contextual_create(bm, geom=[x1, x2, x7, x8]) # Add long side
        bmesh.ops.contextual_create(bm, geom=[x1, x3, x5, x8]) # Add short side
        if args.camera_idx != 6:
            bmesh.ops.contextual_create(bm, geom=[x2, x4, x6, x7]) # Add short side
        if args.camera_idx == 5:
            bmesh.ops.contextual_create(bm, geom=[x3, x4, x5, x6]) # Add long side
    bm.to_mesh(cont_mesh)
    bm.free()
    container = bpy.data.objects.new("Container", cont_mesh.copy())
    collection.objects.link(container)
    container.active_material=mat_container
    # Create mesh for table and assign material
    table_mesh = bpy.data.meshes.new('Table')
    bm = bmesh.new()
    bm.verts.new((0,0,0.099))
    bm.verts.new((0,1,0.099))
    bm.verts.new((1,0,0.099))
    bm.verts.new((1,1,0.099))
    bmesh.ops.contextual_create(bm, geom=bm.verts)
    bm.to_mesh(table_mesh)
    bm.free()
    table = bpy.data.objects.new("Table", table_mesh.copy())
    collection.objects.link(table)
    table.active_material=mat_table
    
def remove_default_cube():
    # Remove default cube
    cube = bpy.data.objects.get("Cube", None)
    if cube is not None:
        bpy.data.objects.remove(cube, do_unlink=True)
    
    
def read_particle_data(csv_file_name, mat_sand, mat_rigid, args=None):
    render_rigids = args is None or not args.hide_rigids
    diameter = 0.002 if args is None else args.diameter 
    # Create new collection for particles
    particle_collection = bpy.data.collections.new('Particles') 
    bpy.context.scene.collection.children.link(particle_collection)
    # Create a mesh for particle
    mesh = bpy.data.meshes.new('Particle')
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=4, diameter=diameter) # d = 0.004
    bm.to_mesh(mesh)
    bm.free()
    # Read particle data, create particle objects and animate them
    # Assumes data to be [id, mat_id, x, y, x]
    # where 
    #     (1) id is the index of the particle
    #     (2) mat_id is the material index: 0 = sand, 1 = rigid body
    #     (3) x,y,z are the coordinates of the particle
    with open(csv_file_name, newline='') as csvfile:
        particle_reader = csv.reader(csvfile, delimiter=',')
        frame = -1
        first_id = None
        for row in particle_reader:
            id, mat_id, x, y, z = int(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4])
            if frame == -1:
                first_id = id
            if id == first_id:
                frame = frame + 1
            if frame == 0 and (render_rigids or mat_id == 0):
                # If at first frame, add particles
                # If hiding rigids do not add them
                basic_sphere = bpy.data.objects.new(f"Particle.{id:03d}", mesh.copy())
                particle_collection.objects.link(basic_sphere)
                particle = bpy.data.objects[f"Particle.{id:03d}"]
                # Add material to the object
                mat = mat_sand if mat_id == 0 else mat_rigid
                particle.active_material=mat

            if render_rigids or mat_id == 0:
                # Set particle location on frame
                particle = bpy.data.objects[f"Particle.{id:03d}"]
                particle.location = x, z, y             # Y and Z axis are the other way around in Blender than in data
                particle.keyframe_insert(data_path="location", frame=frame)
            
def render_animation(output_path, start=1, end=300, step=3, args=None):
    if args is not None and args.save_ffmpeg:
        bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.frame_start = start
    bpy.context.scene.frame_end   = end
    bpy.context.scene.frame_step  = step
    bpy.ops.render.render(animation=True)
   
   
def main(args): 
    if args.camera_position is not None:
        assert len(args.camera_position) == 6, f"Given {len(args.camera_position)} postition arguments when 6 is needed."
        camera_position = [args.camera_position[i] for i in range(0, 3)]
        camera_orientation = [args.camera_position[i] * (3.14/180) for i in range(3, 6)]
    else:
        camera_position = CAMERA_POSITIONS[args.camera_idx][0]
        camera_orientation = CAMERA_POSITIONS[args.camera_idx][1]
    remove_default_cube()
    # Get materials, set camera and background
    colors = MATERIAL_COLORS
    colors['Sand'] = hex_to_rgb(args.sand_color)
    mat_sand, mat_rigid, mat_container, mat_table = get_materials(colors)
    set_camera(position=camera_position, orientation=camera_orientation, resolution=(args.res, args.res), transparent_background=args.use_transparent_background)
    set_background_objects(mat_container, mat_table, args)
    # Read particle data and animate particles
    read_particle_data(args.file_name, mat_sand, mat_rigid, args)
    # Render animation
    render_animation(args.output, start=args.start, end=args.end, step=args.step, args=args)
    print('camera:', args.camera_idx)
    print('transparent background:', args.use_transparent_background)
    print('hide rigids:', args.hide_rigids)
    
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser(description='Render sand simulation result using Blender')
    parser.add_argument('--file_name', required=True, help='File to be visualized. Needs to be .csv file')
    parser.add_argument('--output', required=True, help='Output path')
    parser.add_argument('--start', default=1, type=int, help='First frame to render')
    parser.add_argument('--end', default=300, type=int, help='Last frame to render')
    parser.add_argument('--step', default=3, type=int, help='Frame step')
    parser.add_argument('--res', default=512, type=int, help="Resolution of output images (res, res)")
    parser.add_argument('--camera_idx', default=0, type=int, choices=[0,1,2,3,4,5,6,7], help="Camera position/orientation to be used. 0 = Corner view, 1 = Front view whole, 2 = Front view container, 3 = Top view container, 4 = Back view container, 5 = Top view container rotated, 6 =  Back view angled, 7 = Back view wide")
    parser.add_argument('--camera_position', default=None, nargs='*', type=float, help="Custom camera position and orientation (x_posititon, y_position, z_position, x_rotation, y_rotation, z_rotation)")
    parser.add_argument('--use_transparent_background', action='store_true', help="Render transparent background")
    parser.add_argument('--hide_rigids', action='store_true', help='Hide rigid particles')
    parser.add_argument('--hide_background_objects', action='store_true', help='Hide background objects that is the container and table')
    parser.add_argument('--sand_color', default=0xcc9200, type=auto_int, help='Color of sand')
    parser.add_argument('--diameter', default=0.002, type=float, help='Diameter of particles (d=0.002 -> normal, d=0.004 -> large)')
    parser.add_argument('--save_ffmpeg', action='store_true', help='Save as FFmpeg video instead of individual PNG frames.')
    args = parser.parse_args(argv)
    main(args)