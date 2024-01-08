
import glob
import torch
import cv2
import json
import os
import pyrender
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
from tqdm import tqdm
import trimesh
from icecream import ic
import PIL.Image as pil_img

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

smpl_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 'body_pose': (-1, 69)}
smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}
smplx_shape_except_expression = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3)}

rgb_code ={
    'LightPink': (255,182,193),
    'Pink':(255,192,203),
    'Crimson':(220,20,60),
    'PaleVioletRed':(219,112,147),
    'HotPink':(255,105,180),
    'DeepPink':(255,20,147),
    'MediumVioletRed':(199,21,133),
    'Orchid':(218,112,214),
    'Thistle':(216,191,216),
    'LavenderBlush': (255, 240, 245),
    'plum':(221,160,221),
    'Violet':(238,130,238),
    'Magenta':(255,0,255),
    'Fuchsia':(255,0,255),
    'DarkMagenta':(139,0,139),
    'Purple':(128,0,128),
    'MediumOrchid':(186,85,211),
    'DarkVoilet':(148,0,211),
    'DarkOrchid':(153,50,204),
    'Indigo':(75,0,130),
    'BlueViolet':(138,43,226),
    'MediumPurple':(147,112,219),
    'MediumSlateBlue':(123,104,238),
    'SlateBlue':(106,90,205),
    'DarkSlateBlue':(72,61,139),
    'Lavender':(230,230,250),
    'GhostWhite':(248,248,255),
    'Blue':(0,0,255),
    'MediumBlue':(0,0,205),
    'MidnightBlue':(25,25,112),
    'DarkBlue':(0,0,139),
    'Navy':(0,0,128),
    'RoyalBlue':(65,105,225),
    'CornflowerBlue':(100,149,237),
    'LightSteelBlue':(176,196,222),
    'LightSlateGray':(119,136,153),
    'SlateGray':(112,128,144),
    'DoderBlue':(30,144,255),
    'AliceBlue':(240,248,255),
    'SteelBlue':(70,130,180),
    'LightSkyBlue':(135,206,250),
    'SkyBlue':(135,206,235),
    'DeepSkyBlue':(0,191,255),
    'LightBLue':(173,216,230),
    'PowDerBlue':(176,224,230),
    'CadetBlue':(95,158,160),
    'Azure':(240,255,255),
    'LightCyan':(225,255,255),
    'PaleTurquoise':(175,238,238),
    'Cyan':(0,255,255),
    'DarkTurquoise':(0,206,209),
    'DarkSlateGray':(47,79,79),
    'DarkCyan':	(0,139,139),
    'Teal':(0,128,128),
    'MediumTurquoise':(72,209,204),
    'LightSeaGreen':(32,178,170),
    'Turquoise':(64,224,208),
    'Auqamarin':(127,255,170),
    'MediumAquamarine':(0,250,154),
    'MediumSpringGreen':(245,255,250),
    'MintCream':(0,255,127),
    'SpringGreen':(60,179,113),
    'SeaGreen':	(46,139,87),
    'Honeydew':(240,255,240),
    'LightGreen':(144,238,144),
    'PaleGreen':(152,251,152),
    'DarkSeaGreen':(143,188,143),
    'LimeGreen':(50,205,50),
    'Green':(0,255,0),
    'ForestGreen':(34,139,34),
    'Lime':(0,128,0),
    'DarkGreen':(0,100,0),
    'Chartreuse':(127,255,0),
    'LawnGreen':(124,252,0),
    'GreenYellow':(173,255,47),
    'OliveDrab':(85,107,47),
    'Beige':(107,142,35),
    'LightGoldenrodYellow':(250,250,210),
    'Ivory':(255,255,240),
    'LightYellow':(255,255,224),
    'Yellow':(255,255,0),
    'Olive':(128,128,0),
    'DarkKhaki':(189,183,107),
    'LemonChiffon':(255,250,205),
    'PaleGodenrod':(238,232,170),
    'Khaki':(240,230,140),
    'Gold':(255,215,0),
    'Cornislk':(255,248,220),
    'GoldEnrod':(218,165,32),
    'FloralWhite':(255,250,240),
    'OldLace':(253,245,230),
    'Wheat':(245,222,179),
    'Moccasin':	(255,228,181),
    'Orange':(255,165,0),
    'PapayaWhip':(255,239,213),
    'BlanchedAlmond':(255,235,205),
    'NavajoWhite':(255,222,173),
    'AntiqueWhite':(250,235,215),
    'Tan':(210,180,140),
    'BrulyWood':(222,184,135),
    'Bisque':(255,228,196),
    'DarkOrange':(255,140,0),
    'Linen':(250,240,230),
    'Peru':(205,133,63),
    'PeachPuff':(255,218,185),
    'SandyBrown':(244,164,96),
    'Chocolate':(210,105,30),
    'SaddleBrown':(139,69,19),
    'SeaShell':	(255,245,238),
    'Sienna':(160,82,45),
    'LightSalmon':(255,160,122),
    'Coral':(255,127,80),
    'OrangeRed':(255,69,0),
    'DarkSalmon': (233,150,122),
    'Tomato':(255,99,71),
    'MistyRose':(255,228,225),
    'Salmon':(250,128,114),
    'Snow':(255,250,250),
    'LightCoral':(240,128,128),
    'RosyBrown':(188,143,143),
    'IndianRed':(205,92,92),
    'Red':(255,0,0),
    'Brown':(165,42,42),
    'FireBrick':(178,34,34),
    'DarkRed':(139,0,0),
    'Maroon':(128,0,0),
    'White':(255,255,255),
    'WhiteSmoke':(245,245,245),
    'Gainsboro':(220,220,220),
    'LightGrey':(211,211,211),
    'Silver':(192,192,192),
    'DarkGray':(169,169,169),
    'Gray':(128,128,128),
    'DimGray':(105,105,105),
    'Black':(0,0,0)
}

def render_pose(img, mesh, camera, color_value, return_mask=False, render_front=True):
    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    # render material
    base_color = (1.0, 193/255, 193/255, 1.0)
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0,
            alphaMode='OPAQUE',
            baseColorFactor=base_color)
    
    material_new = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.4,
            alphaMode='OPAQUE',
            emissiveFactor=(0.2, 0.2, 0.2),
            baseColorFactor=(color_value[0]/255, color_value[1]/255, color_value[2]/255, 1))    
    material = material_new
    
    # get body mesh
    body_trimesh = trimesh.load(mesh)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)
    
    aroundy = cv2.Rodrigues(np.array([0, np.radians(-90.), 0]))[0]
    center = body_trimesh.vertices.mean(axis=0)
    rot_vertices = np.dot((body_trimesh.vertices - center), aroundy) + center
    rot_mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(rot_vertices, body_trimesh.faces, process=False), material=material)

    # prepare camera and light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    cam_pose = pyrender2opencv @ np.eye(4)
    
    # build scene
    if render_front:
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)
        scene.add(body_mesh, 'mesh')
    else:
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                                    ambient_light=(0.3, 0.3, 0.3))
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)
        scene.add(rot_mesh, 'mesh')
        
    # render scene
    r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                    viewport_height=img.shape[0],
                                    point_size=1.0)
    
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    
    # 将 color保存为图片
    img_color = pil_img.fromarray((color * 255).astype(np.uint8))
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    img = img / 255
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)
    img = (output_img * 255).astype(np.uint8)

    if return_mask:
        return img, valid_mask, (color * 255).astype(np.uint8)
    
    if render_front:
        return img
    else:
        return img_color

def render_smpler(data_path, output_path, render_front=True):
    seqs = glob.glob(os.path.join(data_path, '**/smplx'), recursive=True)
    seqs = [os.path.dirname(p) for p in seqs]

    for i, seq in enumerate(seqs):
        image_base_path = os.path.join(seq, 'color')
        
        assert os.path.exists(image_base_path)
        
        smplx_path = os.path.join(seq, 'smplx')
        anno_ps = sorted(glob.glob(os.path.join(smplx_path, '*.npz')))

        # group by framestamps
        framestamps = sorted(list(set([os.path.basename(p)[:5] for p in anno_ps 
                                       if 'person' not in os.path.basename(p)]
                                       )))
        for framestamp in tqdm(framestamps, leave=False, desc=f'Seqs {os.path.basename(seq)}'
                               f' : {i}/{len(seqs)}'):
                
            annos = [p for p in anno_ps if framestamp in os.path.basename(p)]
            annos = [p for p in annos if 'person' not in os.path.basename(p)]

            body_model_params = []
            cameras = []
            bbox_sizes = []
            
            image_path = os.path.join(image_base_path, f'{int(framestamp):03d}.png')
            image = cv2.imread(image_path)

            for anno_p in annos:
                anno = dict(np.load(anno_p, allow_pickle=True))
                meta = json.load(open(os.path.join(seq, 'meta', 
                                                os.path.basename(anno_p).replace('.npz', '.json')
                                                )))
                mesh = os.path.join(seq, 'mesh', os.path.basename(anno_p).replace('.npz', '.obj'))

                bbox_size = meta['bbox'][2] * meta['bbox'][3]
                focal_length = meta['focal']
                principal_point = meta['princpt']

                camera = pyrender.camera.IntrinsicsCamera(
                        fx=focal_length[0], fy=focal_length[1],
                        cx=principal_point[0], cy=principal_point[1],)

                # prepare body model params
                intersect_key = list(set(anno.keys()) & set(smplx_shape.keys()))
                body_model_param_tensor = {key: torch.tensor(
                        np.array(anno[key]).reshape(smplx_shape[key]), device=device, dtype=torch.float32)
                                for key in intersect_key if len(anno[key]) > 0}
                
                cameras.append(camera)
                body_model_params.append(body_model_param_tensor)
                bbox_sizes.append(bbox_size)

            # render pose
            bid = bbox_sizes.index(max(bbox_sizes))
            for color, value in rgb_code.items():
                rendered_image = render_pose(img=image,
                                mesh=mesh,
                                camera=cameras[bid],
                                color_value=value,
                                render_front=render_front)           
                
                save_path = os.path.join(output_path, color)
                os.makedirs(save_path, exist_ok=True)

                save_name = os.path.join(save_path, f'{int(framestamp):04d}.png')
                if render_front:
                    cv2.imwrite(save_name, rendered_image)
                else:
                    rendered_image.save(save_name)
   

if __name__ == '__main__':
    data_path = './smplerx'
    output_path = './output'
    render_smpler(data_path=data_path, output_path=output_path, render_front=False)
    