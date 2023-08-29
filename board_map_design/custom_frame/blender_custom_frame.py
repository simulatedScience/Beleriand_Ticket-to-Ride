import bpy
import math

def clear_generated_models():
    # clear all previously generated objects
    for obj in bpy.data.objects:
        if obj.name.startswith("pygen_"):
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.ops.object.delete()

def create_inner_frame(frame_size=(75, 54.2), cylinder_radius=0.2):
    # define the size of the frame
    cylinder_depth = frame_size[0] - 2 * cylinder_radius
    sphere_radius = cylinder_radius

    rx, ry = frame_size[0]/2, frame_size[1]/2
    # add cylinders
    for i, (x, y) in enumerate([(-rx, 0), (0, -ry), (rx, 0), (0, ry)]):
        rotation = (math.pi/2, 0, 0) if i%2==0 else (0, math.pi/2, 0)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=cylinder_radius,
            depth=(frame_size[1] if i % 2 == 0 else frame_size[0]), 
            location=(x,y,0),
            rotation=rotation
        )    
        bpy.context.object.name = "pygen_inner_cylinder_" + str(i)

    # add spheres for the inner frame
    for j, (x, y) in enumerate([(-rx, -ry), (-rx, ry), (rx, -ry), (rx, ry)]):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=sphere_radius, location=(x, y, 0))
        bpy.context.object.name = "pygen_inner_sphere_" + str(j)

    # create a new material
    mat_name = "inner_frame_material"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(name=mat_name)    

    # assign the material to all objects
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            if obj.data.materials:
                # assign to 1st material slot
                obj.data.materials[0] = mat
            else:
                # no slots
                obj.data.materials.append(mat)
                
def create_outer_frame(frame_size=(75, 54.2), outer_border_width=1.6):
    rx, ry = frame_size[0]/2, frame_size[1]/2
    
    # create a new material
    mat_name = "outer_frame_material"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = bpy.data.materials.new(name=mat_name)
        
    # create the outer frame (a box with a hole in the center)
    bpy.ops.mesh.primitive_cube_add(
        size=1, 
        location=(0, 0, -outer_border_width/4)
    )
    outer_frame = bpy.context.object
    outer_frame.dimensions = (
            (outer_border_width + rx) * 2,
            (outer_border_width + ry) * 2,
            outer_border_width/2)
    outer_frame.name = "pygen_outer_frame"
    outer_frame.data.materials.append(mat)

    # create the hole in the center of the outer frame
    bpy.ops.mesh.primitive_cube_add(
        size=1, 
        location=(0, 0, -outer_border_width/2)
    )
    inner_cutout = bpy.context.object
    inner_cutout.dimensions = (rx * 2, ry * 2, outer_border_width)
    inner_cutout.name = "pygen_inner_cutout"
    inner_cutout.data.materials.append(mat)

    # use the boolean modifier to subtract the inner_cutout from the outer_frame
    mod_bool = outer_frame.modifiers.new('Modifier', 'BOOLEAN')
    mod_bool.operation = 'DIFFERENCE'
    mod_bool.object = inner_cutout
    bpy.context.view_layer.objects.active = outer_frame
    bpy.ops.object.modifier_apply({"object": outer_frame}, modifier=mod_bool.name)

    # delete the inner_cutout
    bpy.ops.object.select_all(action='DESELECT')
    inner_cutout.select_set(True)
    bpy.ops.object.delete()
    
                    
def main():
    clear_generated_models()
    
    frame_size = (75, 54.2)
    cylinder_radius = 0.2
    outer_frame_width = 1.9
    
    create_inner_frame(frame_size, cylinder_radius)
    create_outer_frame(frame_size, outer_frame_width)
    
main()

