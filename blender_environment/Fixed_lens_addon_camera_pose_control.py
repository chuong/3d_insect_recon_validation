# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:18:47 2020

@author: cho115
"""

#----------------------------------------------------------------------------------------------------
#   About the plugin
#----------------------------------------------------------------------------------------------------
bl_info = {
    "name" : "Fixed Lens Focus Stacking Photogrammetry (FSP) Setup for small subjects",
    "author" : "Julien Haudegond & Enguerrand De Smet & Chuong Nguyen",
    "description" : "Addon used to setup and render images to make photogrammetry on small subjects",
    "blender" : (2, 80, 0),
    "version": (1,0),
    "location" : "View3D > Sidebar > Photogrammetry",
    "warning" : "",
    "category" : "Photogrammetry",
}


#----------------------------------------------------------------------------------------------------
#   Imports
#----------------------------------------------------------------------------------------------------
import bpy
from bpy.types import Panel, Operator, PropertyGroup

import math
import mathutils
import os
import numpy as np


#----------------------------------------------------------------------------------------------------
#   Practical Functions
#----------------------------------------------------------------------------------------------------
#def local_translate_from_center(obj, translate):
#    local_translate(obj, translate, reset_translation=True)
#
#def local_translate(obj, translate, reset_translation=False):
#    original_translation, original_rotation, original_scale = obj.matrix_world.decompose()
#
#    if reset_translation:
#        original_translation_mat = mathutils.Matrix()
#    else:
#        original_translation_mat = mathutils.Matrix.Translation(original_translation)
#
#    original_rotation_mat = original_rotation.to_matrix().to_4x4()
#    original_scale_mat = mathutils.Matrix.Scale(original_scale[0],4,(1,0,0)) @ mathutils.Matrix.Scale(original_scale[1],4,(0,1,0)) @ mathutils.Matrix.Scale(original_scale[2],4,(0,0,1))
#
#    local_translation_mat = mathutils.Matrix.Translation(translate)
#
#    obj.matrix_world = original_translation_mat @ original_rotation_mat @ local_translation_mat @ original_scale_mat

def focus_translate_from_center(camera, translate):
    focus_translate(camera, translate, reset_translation=True)

def focus_translate(camera, translate, reset_translation=False):
    camera_data = bpy.data.cameras
    if reset_translation:
        camera_data[0].dof.focus_distance = translate
    else:
        camera_data[0].dof.focus_distance += translate
        #print(camera_data[0].dof.focus_distance)

def change_lens_focal_length(camera, lens_focal_length):
    camera_data = bpy.data.cameras
    camera_data[0].lens = lens_focal_length*1000
    #print(camera_data[0].lens, camera_data[0].dof.focus_distance)
    
def sensor_plane_translate(camera, translate):
    camera_data = bpy.data.cameras
    camera_data[0].dof.focus_distance = translate
    print(camera_data[0].lens, camera_data[0].dof.focus_distance)
    
def get_list_cameras(self, context):
    if bpy.data.collections.get("FSP_setup_collection") is None:
        return []

    cameras = []
    for o in bpy.data.collections.get("FSP_setup_collection").objects:
        if o.type == 'CAMERA':
            cameras.append(o)

    return cameras


#----------------------------------------------------------------------------------------------------
#   Base Class
#----------------------------------------------------------------------------------------------------
class FocusStackingPhotogrammetryPanel(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Photogrammetry"


#----------------------------------------------------------------------------------------------------
#   Properties
#----------------------------------------------------------------------------------------------------
class FocusStackingPhotogrammetrySettingsRender(PropertyGroup):
    render_samples_number: bpy.props.IntProperty(
        name="Samples",
        description="Number of samples for rendering",
        default=128,
        min=16,
        max=1024,
        step=16,
    )
    enable_caustics: bpy.props.BoolProperty(
        name="Caustics",
        description="Enable/Disable reflective and refractive caustics. Warning: caustics may cause fireflies issues",
        default=True,
    )

class FocusStackingPhotogrammetrySettingsAcquisition(PropertyGroup):
    step_angle: bpy.props.IntProperty(
        name="Step Angle",
        description="Angle (in degrees) separating the take of two consecutive pictures. If 360, there is no rotation.",
        subtype="ANGLE",
        default=15,
        min=5,
        max=360,
        step=1
    )
    nb_pictures_to_stack: bpy.props.IntProperty(
        name="Stacking Number",
        description="Number of pictures to stack for each camera",
        default=61,
        min=1,
        max=251,
        step=2,
    )
    first_camera: bpy.props.IntProperty(
        name="First Camera",
        description="Number of the first camera to render",
        default=0,
        min=0,
        max=200,
    )
    last_camera: bpy.props.IntProperty(
        name="Last Camera",
        description="Number of the last camera to render",
        default=0,
        min=0,
        max=200,
    )
    ready_for_acquisition: bpy.props.BoolProperty(
        name="Ready for Acquisition",
        default=False,
    )

class FocusStackingPhotogrammetrySettingsCameras(PropertyGroup):
    focal_length: bpy.props.FloatProperty(
        name="Focal Length",
        description="Focal length in millimeters assuming a 36x24mm sensor",
        precision=3,
        subtype="DISTANCE",
        unit="CAMERA",
        default=65,
        min=1,
        max=300,
        step=100,
    )
    
    depth_of_field: bpy.props.BoolProperty(
        name="Depth of Field",
        description="Enable/Disable depth of field effects",
        default=True,
    )
    
    aperture: bpy.props.FloatProperty(
        name="Aperture",
        description="Value of the aperture: F-Stop",
        precision=1,
        default=2.8,
        min=0.1,
        max=128,
        step=10,
    )
    camera_distance: bpy.props.FloatProperty(
        name="Camera Distance to Center",
        description="Set the distance (in meters) between camera and subject's center. This will be considered as the constant focus distance",
        subtype="DISTANCE",
        default=0.2,
        min=0.01,
        max=2,
    )
    backward_translation: bpy.props.FloatProperty(
        name="Backward Translation",
        description="Translate the cameras backward to align the focus point with the front of the subject. Yellow crosses should be aligned with the front of the subject",
        subtype="DISTANCE",
        default=-0.019,
        min=-2,
        max=-0.001,
    )
    forward_translation: bpy.props.FloatProperty(
        name="Forward Translation",
        description="Translate the cameras forward to align the focus point with the back of the subject. Yellow crosses should be aligned with the back of the subject",
        subtype="DISTANCE",
        default=0.024,
        min=0.001,
        max=2,
    )
    angle_between_camera: bpy.props.IntProperty(
        name="Camera Offset Angle",
        description="Angle (in degrees) between two cameras",
        default=15,
        min=1,
        max=180,
    )


#----------------------------------------------------------------------------------------------------
#   Main Panel
#----------------------------------------------------------------------------------------------------
class FSP_PT_main(FocusStackingPhotogrammetryPanel):
    bl_label = 'Fixed Lens Focus Stacking Photogrammetry (FLFSP)'

    def draw(self, context):
        self.layout.label(text="Create a setup for small subjects photogrammetry.")


#----------------------------------------------------------------------------------------------------
#   Global Options Sub-Panel
#----------------------------------------------------------------------------------------------------
def clean_scene(self, context):
    for o in bpy.data.objects:
        bpy.data.objects.remove(o, do_unlink=True)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c, do_unlink=True)
    for l in bpy.data.lights:
        bpy.data.lights.remove(l, do_unlink=True)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c, do_unlink=True)

    context.scene.fsp_settings_acquisition.ready_for_acquisition = False

def clean_setup(self, context):
    # Delete objects in our collection
    if not bpy.data.collections.get("FSP_setup_collection") is None:
        for o in bpy.data.collections.get("FSP_setup_collection").objects:
            bpy.data.objects.remove(o, do_unlink=True)

    context.scene.fsp_settings_acquisition.ready_for_acquisition = False


class FSP_OT_clean_scene(Operator):
    bl_idname = "fsp.clean_scene"
    bl_label = "Clean The Scene"
    bl_description = "Clean completely the scene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        clean_scene(self, context)

        return {'FINISHED'}

class FSP_OT_clean_setup(Operator):
    bl_idname = "fsp.clean_setup"
    bl_label = "Clean The Setup (only)"
    bl_description = "Clean only the collection containing the setup"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        clean_setup(self, context)

        return {'FINISHED'}

class FSP_OT_setup_scene(Operator):
    bl_idname = "fsp.setup_scene"
    bl_label = "Setup The Scene"
    bl_description = "Create the setup for small subjects renders"
    bl_options = {'REGISTER', 'UNDO'}

    nb_camera: bpy.props.IntProperty(
        name="Camera Number",
        description="Number of cameras (should be an odd number)",
        default=13,
        min=1,
        max=31,
        step=2,
    )
    angle_between_camera: bpy.props.IntProperty(
        name="Camera Offset Angle",
        description="Angle (in degrees) between two cameras",
        default=15,
        min=1,
        max=180,
    )
    camera_distance: bpy.props.FloatProperty(
        name="Camera Distance to Center",
        description="Set the distance (in meters) between camera and subject's center. This will be considered as the constant focus distance",
        subtype="DISTANCE",
        default=0.2,
        min=0.01,
        max=2,
    )
    backward_translation: bpy.props.FloatProperty(
        name="Backward Translation",
        description="Translate the cameras backward to align the focus point with the front of the subject. Yellow crosses should be aligned with the front of the subject",
        subtype="DISTANCE",
        default=-0.019,
        min=-2,
        max=-0.001,
    )
    forward_translation: bpy.props.FloatProperty(
        name="Forward Translation",
        description="Translate the cameras forward to align the focus point with the back of the subject. Yellow crosses should be aligned with the back of the subject",
        subtype="DISTANCE",
        default=0.024,
        min=0.001,
        max=2,
    )
    align_preview: bpy.props.EnumProperty(
        name='Preview Camera Focus Position',
        description='Choose the camera position you want to preview',
        items={
            ('Backward', 'Backward', 'Preview the camera position when the focus point is aligned with the front of the subject'),
            ('Center', 'Center', 'Preview the camera position when the focus point is aligned with the center of the subject'),
            ('Forward', 'Forward', 'Preview the camera position when the focus point is aligned with the back of the subject'),
        },
        default='Center',
    )

    def execute(self, context):
        settings_cameras = context.scene.fsp_settings_cameras
        settings_acquisition = context.scene.fsp_settings_acquisition

        clean_setup(self, context)  # If the setup already exist, we need to clean it
        self.__create_collection(context)

        # Focus alignment preview
        if self.align_preview == "Backward":
            camera_align = self.backward_translation
        elif self.align_preview == "Center":
            camera_align = 0
        else:
            camera_align = self.forward_translation

        # Empty object to contain the cameras and make them rotate easily
        cameras_group = bpy.data.objects.new('FSP_cameras_group', None)
        cameras_group.location = (0,0,0)
        cameras_group.empty_display_type = 'PLAIN_AXES'
        self.__linkToCollection(cameras_group)

        # Create ONE camera data
        camera = self.__create_camera(context)
#        camera.dof.focus_distance = self.camera_distance
        camera.dof.focus_distance = self.camera_distance + camera_align

        angle_first_camera = 90.0 - int(self.nb_camera/2) * self.angle_between_camera

        for i in range(self.nb_camera):
            angle_current_camera = angle_first_camera + i * self.angle_between_camera
#            camera_object = self.__create_camera_object(context, f"FSP_camera_{str(i).zfill(3)}", camera, (0.0, 0.0, self.camera_distance + camera_align), (angle_current_camera, 0.0, 90.0))
            camera_object = self.__create_camera_object(context, f"FSP_camera_{str(i).zfill(3)}", camera, (0.0, 0.0, self.camera_distance), (angle_current_camera, 0.0, 90.0))
            camera_object.parent = cameras_group

        # Update global settings we defined
        settings_cameras.camera_distance = self.camera_distance
        settings_cameras.backward_translation = self.backward_translation
        settings_cameras.forward_translation = self.forward_translation
        #settings_cameras.angle_between_camera = self.angle_between_camera
        settings_acquisition.last_camera = self.nb_camera - 1

        settings_acquisition.ready_for_acquisition = True

        return {'FINISHED'}

    def __create_collection(self, context):
        if bpy.data.collections.get("FSP_setup_collection") is None:
            FSP_setup_collection = bpy.data.collections.new("FSP_setup_collection")
            context.scene.collection.children.link(FSP_setup_collection)

    def __linkToCollection(self, obj):
        bpy.data.collections['FSP_setup_collection'].objects.link(obj)

    def __create_camera(self, context, name="FSP_camera"):
        settings_cameras = context.scene.fsp_settings_cameras

        # Update or create the data camera
        try:
            camera = bpy.data.cameras[name]
        except:
            camera = bpy.data.cameras.new(name)

        # Basic settings
        camera.type = 'PERSP'
        camera.lens_unit = 'MILLIMETERS'
        camera.lens = settings_cameras.focal_length

        camera.sensor_fit = 'HORIZONTAL'
        camera.sensor_width = 36
        camera.sensor_height = 24

        camera.clip_start = 0.005
        camera.clip_end = 10
        camera.dof.use_dof = settings_cameras.depth_of_field
        camera.dof.aperture_fstop = settings_cameras.aperture
        camera.dof.focus_distance = settings_cameras.camera_distance

        camera.show_limits = True
        camera.display_size = 0.02

        return camera

    def __create_camera_object(self, context, name, camera, loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)):
        radians_rot = tuple([math.radians(a) for a in rot])  # Convert angles to radians
        obj = bpy.data.objects.new(name, camera)

        translation_mat = mathutils.Matrix.Translation(loc)

        rotation_mat_x = mathutils.Matrix.Rotation(math.radians(rot[0]), 4, 'X')
        rotation_mat_y = mathutils.Matrix.Rotation(math.radians(rot[1]), 4, 'Y')
        rotation_mat_z = mathutils.Matrix.Rotation(math.radians(rot[2]), 4, 'Z')
        rotation_mat = rotation_mat_z @ rotation_mat_y @ rotation_mat_x  # Euler XYZ

        scale_mat = mathutils.Matrix()

        obj.matrix_world = rotation_mat @ translation_mat @ scale_mat

        self.__linkToCollection(obj)

        return obj

class FSP_PT_main_global(FocusStackingPhotogrammetryPanel):
    bl_label = 'Global Options'
    bl_parent_id = "FSP_PT_main"

    def draw(self, context):
        layout = self.layout

        row = layout.row(align=True)
        row.operator('fsp.clean_scene')
        row.operator('fsp.clean_setup')

        row = layout.row()
        row.operator('fsp.setup_scene')


#----------------------------------------------------------------------------------------------------
#   Settings Sub-Panel
#----------------------------------------------------------------------------------------------------
def move_cameras_to_extrem(self, context, extrem):
    settings_cameras = context.scene.fsp_settings_cameras

    list_cameras = get_list_cameras(self, context)

    if extrem == "Backward":
        z_translate = settings_cameras.camera_distance + settings_cameras.backward_translation
    elif extrem == "Center":
        z_translate = settings_cameras.camera_distance
    elif extrem == "Forward":
        z_translate = settings_cameras.camera_distance + settings_cameras.forward_translation
    else:
        return

#    translation = (0.0, 0.0, z_translate)

    for camera in list_cameras:
#        local_translate_from_center(camera, translation)
        focus_translate_from_center(camera, z_translate)


def apply_settings_render(self, context):
    # Useful variables
    rdr = context.scene.render
    egn = context.scene.cycles  # engine
    settings_render = context.scene.fsp_settings_render

    rdr.engine = 'CYCLES'
    egn.device = 'GPU'
    rdr.tile_x = 256
    rdr.tile_y = 256
    egn.samples = settings_render.render_samples_number
    egn.caustics_reflective = settings_render.enable_caustics
    egn.caustics_refractive = settings_render.enable_caustics

    context.space_data.clip_start = 0.001  # Avoid clipping in the viewer

def apply_settings_cameras(self, context):
    if bpy.data.collections.get("FSP_setup_collection") is None:
        return

    already_set = []
    for o in bpy.data.collections.get("FSP_setup_collection").objects:
        if o.type == 'CAMERA':
            camera = o.data

            # Avoid to set again already set cameras
            if camera in already_set:
                continue

            camera.dof.aperture_fstop = context.scene.fsp_settings_cameras.aperture
            camera.lens = context.scene.fsp_settings_cameras.focal_length

            # Mark the camera as set
            already_set.append(camera)
            
def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def fix_pose(pose_matrix_array):

    R_bcam2cv = np.array([[1,0,0],[0,0,1],[0,-1,0]])

    i = pose_matrix_array.shape[0]
    R = pose_matrix_array[:,0:3,0:3]
    T = pose_matrix_array[::,0:3,3]
    T = np.resize(T, (i,3,1))


    R_world2bcam = R
    T_world2bcam = T

    R_world2cv = np.matmul(-1*R_bcam2cv, R_world2bcam)
    T_world2cv = np.matmul(R_bcam2cv, T_world2bcam)

    R_world2cv[:,:,0] = -R_world2cv[:,:,0]
    
    RT = np.append(R_world2cv, T_world2cv, axis=2)

    return RT


class FSP_OT_settings_render_apply(Operator):
    bl_idname = "fsp.settings_render_apply"
    bl_label = "Apply Settings"
    bl_description = "Apply those settings to the render"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        apply_settings_render(self, context)

        return {'FINISHED'}

class FSP_OT_settings_cameras_apply(Operator):
    bl_idname = "fsp.settings_cameras_apply"
    bl_label = "Apply Settings"
    bl_description = "Apply those settings to the cameras"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        apply_settings_cameras(self, context)

        return {'FINISHED'}

class FSP_OT_preview_backward(Operator):
    bl_idname = "fsp.preview_backward"
    bl_label = "Preview Camera Position Backward"
    bl_description = "Translate the cameras to align the focus point with the front of the subject"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        move_cameras_to_extrem(self, context, "Backward")

        return {'FINISHED'}

class FSP_OT_preview_center(Operator):
    bl_idname = "fsp.preview_center"
    bl_label = "Preview Camera Position Center"
    bl_description = "Translate the cameras to align the focus point with the center of the subject"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        move_cameras_to_extrem(self, context, "Center")

        return {'FINISHED'}

class FSP_OT_preview_forward(Operator):
    bl_idname = "fsp.preview_forward"
    bl_label = "Preview Camera Position Forward"
    bl_description = "Translate the cameras to align the focus point with the back of the subject"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        move_cameras_to_extrem(self, context, "Forward")

        return {'FINISHED'}

class FSP_PT_main_settings(FocusStackingPhotogrammetryPanel):
    bl_label = 'Settings'
    bl_parent_id = "FSP_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        pass

class FSP_PT_main_settings_render(FocusStackingPhotogrammetryPanel):
    bl_label = 'Render'
    bl_parent_id = "FSP_PT_main_settings"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation

        rdr = context.scene.render
        settings_render = context.scene.fsp_settings_render

        layout.label(text="Render Engine will be set to 'Cycles' and device will be GPU.")
        layout.label(text="Tile Size will be 256x256 (a good value for GPU rendering).")

        col = layout.column(align=True)
        col.prop(rdr, "resolution_x", text="Resolution X")
        col.prop(rdr, "resolution_y", text="Y")
        col.prop(rdr, "resolution_percentage", text="%")

        col = layout.column(align=True)
        col.prop(settings_render, "render_samples_number", text="Render Samples")
        col.prop(settings_render, "enable_caustics", text="Caustics")

        row = layout.row()
        row.operator('fsp.settings_render_apply')

class FSP_PT_main_settings_acquisition(FocusStackingPhotogrammetryPanel):
    bl_label = 'Acquisition'
    bl_parent_id = "FSP_PT_main_settings"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation

        settings_acquisition = context.scene.fsp_settings_acquisition

        col = layout.column()
        col.prop(settings_acquisition, "step_angle")
        col.prop(settings_acquisition, "nb_pictures_to_stack")

class FSP_PT_main_settings_cameras(FocusStackingPhotogrammetryPanel):
    bl_label = 'Cameras'
    bl_parent_id = "FSP_PT_main_settings"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation

        settings_cameras = context.scene.fsp_settings_cameras

        col = layout.column()
        col.prop(settings_cameras, "focal_length")
        col.prop(settings_cameras, "aperture")
        col.prop(settings_cameras, "depth_of_field", text="Depth of Field")
        col.operator("fsp.settings_cameras_apply")

        layout.label(text="Preview Camera Position:")
        row = layout.row(align=True)
        row.operator("fsp.preview_backward", text="Backward")
        row.operator("fsp.preview_center", text="Center")
        row.operator("fsp.preview_forward", text="Forward")


#----------------------------------------------------------------------------------------------------
#   Rendering Sub-Panel
#----------------------------------------------------------------------------------------------------
class FSP_OT_render(Operator):
    bl_idname = "fsp.render"
    bl_label = "Render"
    bl_description = "Render the selected cameras"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):

        if not bpy.data.filepath:
            print("You must save the Blender Project in a specific folder before rendering. An image folder will be created next to the Blender Project file.")
            return {'FINISHED'}

        # Get the img folder path
        file_path = bpy.data.filepath
        cur_dir = os.path.dirname(file_path)
        img_dir = os.path.join(cur_dir, "img")
        # Create the img folder if it does not exist
        os.makedirs(img_dir, exist_ok=True)

        # Get Group and list of cameras
        cameras_group = bpy.context.scene.objects['FSP_cameras_group']
        list_cameras = get_list_cameras(self, context)
        #print(len(list_cameras))

        # Set the Group to the initial rotation angle
        cameras_group.rotation_euler[2] = 0

        # Different settings
        first_camera_index = context.scene.fsp_settings_acquisition.first_camera
        last_camera_index = context.scene.fsp_settings_acquisition.last_camera
        acquisition_step_angle = context.scene.fsp_settings_acquisition.step_angle
        stacking_number = context.scene.fsp_settings_acquisition.nb_pictures_to_stack
        backward_translation = context.scene.fsp_settings_cameras.backward_translation
        forward_translation = context.scene.fsp_settings_cameras.forward_translation
        tilt_angle = context.scene.fsp_settings_cameras.angle_between_camera
        dof = context.scene.fsp_settings_cameras.depth_of_field
        #print(tilt_angle)
        
        average_lens_focal_length = context.scene.fsp_settings_cameras.focal_length/1000
        average_focus_distance = context.scene.fsp_settings_cameras.camera_distance
        
        max_focus_distance = average_focus_distance + forward_translation 
        min_focus_distance = average_focus_distance + backward_translation 
        #print(average_focus_distance, max_focus_distance, min_focus_distance)
        
        max_camera_focal_length = min_focus_distance*average_lens_focal_length/(min_focus_distance-average_lens_focal_length)
        min_camera_focal_length = max_focus_distance*average_lens_focal_length/(max_focus_distance-average_lens_focal_length)
        #print(max_camera_focal_length, min_camera_focal_length)
        
        translate_camera_focal_length=0
        if stacking_number>1:
            translate_camera_focal_length = (max_camera_focal_length - min_camera_focal_length) / (stacking_number-1)
        
        initial_orientation=np.array([[1,0,0],[0,1,0],[0,0,1]])
        initial_position=np.array([0,0,-10])
        positions=np.zeros([len(list_cameras)*2,3])
        m=0
        
        for i in range(len(list_cameras)):
            for j in range(2): 
                beta=math.radians(acquisition_step_angle*j)
                R=np.array([[math.cos(beta),0.0,math.sin(beta)],[0.00,1,0.00],[-math.sin(beta),0.00,math.cos(beta)]])
                alpha=math.radians(tilt_angle*(-i+int(len(list_cameras)/2)))
                R1=np.array([[1,0,0],[0,math.cos(alpha),-math.sin(alpha)],[0,math.sin(alpha),math.cos(alpha)]])
                R_z=np.array([[math.cos(0),-math.sin(0),0],[math.sin(0),math.cos(0),0],[0,0,1]])
                R_combined=np.dot(initial_orientation,np.dot(R_z,np.dot(R,R1)))
                
                positions[m,:]=np.dot(R_combined,initial_position)
                m=m+1
                
        ang=np.zeros([len(list_cameras)])

        for i in range(len(list_cameras)):
                
            x1, y1, z1 = positions[i*2,0], positions[i*2,2], positions[i*2,1]
            x2, y2, z2 = positions[i*2+1,0], positions[i*2+1,2], positions[i*2+1,1]
            
            ang[i] = math.degrees(math.acos( (x1*x2 + y1*y2 + z1*z1) / np.sqrt( (x1*x1 + y1*y1 + z1*z1)*(x2*x2+y2*y2+z2*z2) ) ))
        
        #print(ang)

        print("---------- Rendering start ----------")
        
        Ts = []
        
        for camera in list_cameras:
            # Check if the camera is in the range defined by the user
            camera_index = int(camera.name.split("FSP_camera_")[-1])
            if camera_index < first_camera_index or camera_index > last_camera_index:
                continue

            # Set the current scene camera to this one to make renders
            context.scene.camera = camera

            # Create the img folder for this camera if it does not exist
            #camera_dir = os.path.join(img_dir, camera.name)
            camera_dir = os.path.join(img_dir, 'x='+str(int(camera.name[11:14])))
            os.makedirs(camera_dir, exist_ok=True)
            #print(int(camera.name[13]),camera.name[13])
            print(ang[int(camera.name[11:14])])
            if ang[int(camera.name[11:14])]!=0:
                modified_acquisition_step_angle = ang[int(len(list_cameras)/2)]*acquisition_step_angle/ang[int(camera.name[11:14])]
            if ang[int(camera.name[11:14])]==0:
                modified_acquisition_step_angle=360
            pan_angle_pos=int(round((360/modified_acquisition_step_angle)))
            modified_acquisition_step_angle=360/pan_angle_pos
            print('modified_pan_step_angle: ',modified_acquisition_step_angle)

            for rotation_step in range(pan_angle_pos):
                cameras_group.rotation_euler[2] = math.radians(rotation_step * modified_acquisition_step_angle) # Rotate the group on each step of the process

                # Place the camera in the space
                move_cameras_to_extrem(self, context, "Backward")
                
                # Create the img folder for this camera rotation if it does not exist
                camera_rotation_dir = os.path.join(camera_dir, 'y='+str(rotation_step))
                os.makedirs(camera_rotation_dir, exist_ok=True)
                
                if dof == False:
                    translate_step = stacking_number//2
                    current_camera_focal_length = max_camera_focal_length - translate_step * translate_camera_focal_length
                    current_focus_distance = current_camera_focal_length*average_lens_focal_length/(current_camera_focal_length-average_lens_focal_length)
                    
                    change_lens_focal_length(camera, current_camera_focal_length)
                    sensor_plane_translate(camera, current_focus_distance)
                    
                    self.__startRender(camera_rotation_dir, f"{camera.name}_angle_{str(rotation_step * modified_acquisition_step_angle).zfill(3)}_focus_{str(translate_step).zfill(3)}")
                # Translate the camera from back to front
                else:
                    for translate_step in range(stacking_number):
#                       local_translate(camera, (0.0, 0.0, -translation_to_subject))
                        current_camera_focal_length = max_camera_focal_length - translate_step * translate_camera_focal_length
                        current_focus_distance = current_camera_focal_length*average_lens_focal_length/(current_camera_focal_length-average_lens_focal_length)
                        #print(translate_step, current_camera_focal_length, current_focus_distance)
                    
                        change_lens_focal_length(camera, current_camera_focal_length)
                        sensor_plane_translate(camera, current_focus_distance)

                        self.__startRender(camera_rotation_dir, f"{camera.name}_angle_{str(rotation_step * modified_acquisition_step_angle).zfill(3)}_focus_{str(translate_step).zfill(3)}")
                        
                print(camera.matrix_world)
                T_matrix = listify_matrix(camera.matrix_world)
                Ts.append(T_matrix)

        # Set the Group to the initial rotation angle
        cameras_group.rotation_euler[2] = 0
        Ts = np.array(Ts)
        
        RT = fix_pose(Ts)
        
        np.save(img_dir,RT)
        
        return {'FINISHED'}

    def __startRender(self, img_dir, img_name):
        bpy.context.scene.render.filepath = os.path.join(img_dir, img_name)
        bpy.ops.render.render(write_still=True)


class FSP_PT_main_rendering(FocusStackingPhotogrammetryPanel):
    bl_label = 'Rendering'
    bl_parent_id = "FSP_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings_acquisition = context.scene.fsp_settings_acquisition

        layout.enabled = settings_acquisition.ready_for_acquisition

        row = layout.row(align=True)
        row.prop(settings_acquisition, "first_camera")
        row.prop(settings_acquisition, "last_camera")

        if not bpy.data.filepath:
            layout.label(text="You must save the Blender Project before rendering.")
            row = layout.row()
            row.enabled = False
            row.operator('fsp.render')
        else:
            row = layout.row()
            row.operator('fsp.render')


classes = (
    FocusStackingPhotogrammetrySettingsRender,
    FocusStackingPhotogrammetrySettingsAcquisition,
    FocusStackingPhotogrammetrySettingsCameras,
    FSP_PT_main,
    FSP_OT_clean_scene,
    FSP_OT_clean_setup,
    FSP_OT_setup_scene,
    FSP_PT_main_global,
    FSP_OT_settings_render_apply,
    FSP_OT_settings_cameras_apply,
    FSP_OT_preview_backward,
    FSP_OT_preview_center,
    FSP_OT_preview_forward,
    FSP_PT_main_settings,
    FSP_PT_main_settings_render,
    FSP_PT_main_settings_acquisition,
    FSP_PT_main_settings_cameras,
    FSP_OT_render,
    FSP_PT_main_rendering,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    # Group properties
    bpy.types.Scene.fsp_settings_render = bpy.props.PointerProperty(type=FocusStackingPhotogrammetrySettingsRender)
    bpy.types.Scene.fsp_settings_acquisition = bpy.props.PointerProperty(type=FocusStackingPhotogrammetrySettingsAcquisition)
    bpy.types.Scene.fsp_settings_cameras = bpy.props.PointerProperty(type=FocusStackingPhotogrammetrySettingsCameras)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    # Group properties
    del bpy.types.Scene.fsp_settings_render
    del bpy.types.Scene.fsp_settings_acquisition
    del bpy.types.Scene.fsp_settings_cameras

if __name__ == "__main__":
    register()
