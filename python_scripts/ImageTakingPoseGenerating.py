import bpy
import math
import json

import bpy
import bpy_extras
import numpy as np
from mathutils import Matrix
from mathutils import Vector
from math import radians


distance = -50
objName = "Melitaea_britomartis_sf"
output_dir = 'E:/ENGN4200/InsectModel/beechnut-fagus-sylvatica/full_images/'
RESULTS_PATH = '1080images'
fp = bpy.path.abspath(f"//{RESULTS_PATH}")
print(fp)
VIEWS = 334
stepsize = 360.0 / VIEWS
RESOLUTION = 1080


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def initCL():
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1,1,1,1)
    camera_data = bpy.data.cameras.new('camera')
    cam = bpy.data.objects.new("camera", camera_data)
    
    rx = 90*math.pi/180
    ry = 0*math.pi/180
    rz = 0*math.pi/180
    
    cam.location = [0,distance,0]
    cam.rotation_euler = [rx,ry,rz]
    
    bpy.context.collection.objects.link(cam)
    bpy.context.view_layer.objects.active = cam
    bpy.context.scene.camera = bpy.data.objects['camera']
    
    frontlight_data = bpy.data.lights.new(name = "frontlight", type = 'POINT')
    frontlight = bpy.data.objects.new(name="frontlight", object_data=frontlight_data)
    
    frontlight.location = [0, distance, 10]
    frontlight.rotation_euler = [rx,ry,rz]
    
    bpy.context.collection.objects.link(frontlight)
    bpy.context.view_layer.objects.active = frontlight
    

  
def deleteThing():
    bpy.data.objects['camera'].select_set(True)
    bpy.data.objects['frontlight'].select_set(True)
    bpy.ops.object.delete()
    

def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()
    
def imageTaking():
    obj_camera = bpy.data.objects["camera"]
    obj_other = bpy.data.objects[objName]
    output_name = 'hight %d rotate %d.jpg'
    P = []
    numID = 0
    out_data = {
        'camera_angle_x': obj_camera.data.angle_x,
    }
    out_data['frames'] = []

    for j in range (-90,90,10):
        deg = (90-abs(j)) / 10 * 4
        if deg == 0:
            deg = 1
        gap = int(360//deg)
        height = -distance * math.sin(-j*math.pi/180)
        y_dist = math.sqrt(distance**2 - height**2)
        obj_camera.location = (0,y_dist,height)
        bpy.context.view_layer.update()
        look_at(obj_camera, obj_other.matrix_world.to_translation())
        for i in range(0,360,gap):
            y = y_dist * math.cos(i*math.pi/180)
            x = y_dist * math.sin(i*math.pi/180)
            obj_camera.location = (x, y, height)
            bpy.context.view_layer.update()
            look_at(obj_camera, obj_other.matrix_world.to_translation())
            
            bpy.context.scene.render.filepath = fp + '/' + str(numID)
            bpy.ops.render.render(write_still = True)

            numID += 1

    obj_camera.location = (0,0,distance)
    bpy.data.objects[objName].rotation_euler = [0, 0, 0]
    bpy.context.view_layer.update()
    look_at(obj_camera, obj_other.matrix_world.to_translation())
    bpy.context.scene.render.filepath = fp + '/' + str(numID)
    bpy.ops.render.render(write_still = True)

          
if __name__ == "__main__":
    
    initCL()
    imageTaking()