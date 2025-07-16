import bpy
import bmesh
import sys
import os

# Parse arguments
argv = sys.argv
if "--" not in argv:
    print("Usage: blender --background --python blender_single_decimation.py -- <input_file> <output_file> [target_vertices]")
    sys.exit(1)

idx = argv.index("--")
if len(argv) < idx + 3:
    print("Error: Missing required arguments")
    sys.exit(1)

input_file = os.path.abspath(argv[idx + 1])
output_file = os.path.abspath(argv[idx + 2])
target_vertices = int(argv[idx + 3]) if len(argv) > idx + 3 else 40000

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

EXTS = {'.obj': 'OBJ', '.fbx': 'FBX', '.glb': 'GLTF', '.gltf': 'GLTF'}

def clear_scene():
    """Clear all objects and data from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)
    for img in bpy.data.images:
        bpy.data.images.remove(img)

def decimate_mesh(input_path, output_path, target_vtx):
    """Decimate a single mesh file"""
    name, ext = os.path.splitext(os.path.basename(input_path))
    fmt = EXTS.get(ext.lower())
    
    if not fmt:
        print(f"Error: Unsupported file format {ext}")
        return False
    
    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}")
        return False
    
    print(f"Processing: {input_file} -> {output_file}")
    print(f"Target vertices: {target_vtx}")
    
    clear_scene()
    
    try:
        # Import mesh
        if fmt == 'OBJ':
            bpy.ops.wm.obj_import(filepath=input_path)
        elif fmt == 'FBX':
            bpy.ops.import_scene.fbx(filepath=input_path)
        else:  # GLTF/GLB
            bpy.ops.import_scene.gltf(filepath=input_path)
        
        # Get all mesh objects
        meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        if not meshes:
            print("Error: No mesh objects found in file")
            return False
        
        # Join all meshes into one
        bpy.ops.object.select_all(action='DESELECT')
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        
        if len(meshes) > 1:
            bpy.ops.object.join()
        
        obj = bpy.context.view_layer.objects.active
        me = obj.data
        
        print(f"Original vertex count: {len(me.vertices)}")
        
        # Repair mesh: remove doubles and fill holes
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        
        # Fill holes
        holes = [e for e in bm.edges if e.is_boundary]
        if holes:
            bmesh.ops.holes_fill(bm, edges=holes, sides=0)
        
        bm.normal_update()
        bm.to_mesh(me)
        bm.free()
        
        # Update normals and smoothing
        me.calc_normals_split()
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = 3.14159
        bpy.ops.object.shade_smooth()
        
        # Decimate if necessary
        vcount = len(me.vertices)
        if vcount > target_vtx:
            print(f"Decimating from {vcount} to {target_vtx} vertices")
            mod = obj.modifiers.new("Decimate", "DECIMATE")
            mod.decimate_type = 'COLLAPSE'
            
            try:
                mod.vertex_count = target_vtx
            except AttributeError:
                mod.ratio = target_vtx / vcount
            
            bpy.ops.object.modifier_apply(modifier=mod.name)
            me.calc_normals_split()
            bpy.ops.object.shade_smooth()
            
            final_count = len(me.vertices)
            print(f"Final vertex count: {final_count}")
        else:
            print(f"Mesh already has {vcount} vertices, skipping decimation")
        
        # Export mesh
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        
        # Export as GLB
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format='GLB',
            use_selection=True,
            export_apply=True
        )
        
        print(f"Successfully exported: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing mesh: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run decimation
success = decimate_mesh(input_file, output_file, target_vertices)
sys.exit(0 if success else 1)
