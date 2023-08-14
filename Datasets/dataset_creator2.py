
#from matplotlib import pyplot as plt
import os, cv2, colorsys, copy, argparse, importlib
import numpy as np    ## if you haven't got it, just "pip3 install numpy"
import open3d as o3d  ## if you haven't got it, just "pip3 install open3d"
from tqdm import tqdm ## if you haven't got it, just "pip3 install tqdm"
#from importlib import reload
import time


## General Params
img_width, img_height = 1000, 1000
min_obj_size, max_obj_size = 10, 250
max_fails = 20
max_area_overlap = 0.8

## Rendering Params
render_res = 400
center = [0, 0, 0]  # look_at target
rpy_amp    = 10         # rotation span (for randomly drawing)

# colors global variables
base_rgb_colors = np.array([
    [  0,   0, 255], # blue
    [255,   0,   0], # red
    [255, 255,   0], # yellow
    [140,   0, 255], # purple
    [  0,   0,   0], # black
    [  0, 255,   0]  # green
], dtype=np.uint8)
hsv_offsets = np.array([
    [230/360, 0.88, 0.5], # blue
    [-13/360, 0.88, 0.5], # red
    [ 45/360,  0.7, 0.5], # yellow
    [257/360,  0.7, 0.5], # purple
    [      0,    0,   0], # black
    [ 99/360,  0.7, 0.5]  # green
], dtype=np.float32)
hsv_scalers = np.array([
    [ 9/360, 0.12,  0.5], # blue
    [21/360, 0.12,  0.5], # red
    [21/360,  0.3,  0.5], # yellow
    [22/360,  0.3,  0.5], # purple
    [    1,   0.1, 0.05], # black
    [61/360,  0.3,  0.5]  # green
], dtype=np.float32)

def create_renderer():
    render = o3d.visualization.rendering.OffscreenRenderer(render_res, render_res)
    render.scene.scene.enable_sun_light(False)
    render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
    render.scene.set_background([0,0,0,0])

    v_fov      = 15.0       # vertical field of view: between 5 and 90 degrees
    near_plane = 0.1
    far_plane  = 1e3
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    render.scene.camera.set_projection(v_fov, 1, near_plane, far_plane, fov_type)
    eye = [0, 0, 90]  # camera position
    up = [0, 1, 0]  # camera orientation
    render.scene.camera.look_at(center, eye, up)

    #global add_geometry_count
    #add_geometry_count = 0

    global meshes
    meshes = [o3d.io.read_triangle_mesh("3D_models/Risiko_Tank_v3.stl"), o3d.io.read_triangle_mesh("3D_models/RisikoFlag_V2.stl")]

    return render

def create_mtl(rgba_color:np.ndarray):
    mtl = o3d.visualization.rendering.MaterialRecord()    # or MaterialRecord(), for later versions of Open3D
    mtl.base_color = rgba_color  # RGBA
    mtl.shader = "defaultLit"
    return mtl

def generate_color_using_hsv(obj_class:int) -> np.ndarray:
    color = np.random.uniform(size=3).astype(np.float32)
    color = color * hsv_scalers[obj_class % 6] + hsv_offsets[obj_class % 6]
    rgba_color = np.ones(4, dtype=np.float32)
    rgba_color[:3] = colorsys.hsv_to_rgb(color[0], color[1], color[2])
    return rgba_color

def create_obj(obj_class:int, rescaled_size:int) -> tuple[np.ndarray, np.ndarray]:
    '''
    Function renders model of object specified by class in a random "pose" (rotated on all three axis in a random way)
    Returns a tuple containing 2 images:
        - [0] -> actual rgb image of the object with white background
        - [1] -> mask of the first image useful to paste the object alone without the background
    '''

    color = generate_color_using_hsv(obj_class)
    mtl = create_mtl(color)

    roll, pitch, yaw = np.random.rand()*rpy_amp, np.random.rand()*rpy_amp, np.random.rand()*rpy_amp

    mesh_r = copy.deepcopy(meshes[obj_class // 6])
    R = mesh_r.get_rotation_matrix_from_xyz((roll, pitch, yaw))
    mesh_r.rotate(R, center=center)

    render.scene.add_geometry("rotated_model", mesh_r, mtl)
    #global add_geometry_count
    #add_geometry_count = add_geometry_count + 1

    # Read the image into a variable
    rendered_img = np.array(render.render_to_image())
    render.scene.remove_geometry("rotated_model")

    rendered_img = rendered_img.astype(np.uint8)
    rendered_img = cv2.resize(rendered_img, [rescaled_size, rescaled_size], interpolation=cv2.INTER_LINEAR)
    imgray = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2GRAY)
    
    min_vals = np.max(imgray, axis=1)
    x1 = next((x for x in range(rescaled_size) if min_vals[x] > 1), None)
    x2 = next((x for x in range(rescaled_size-1, 0, -1) if min_vals[x] > 1), None) + 1
    np.max(imgray, axis=0, out=min_vals)
    y1 = next((y for y in range(rescaled_size) if min_vals[y] > 1), None)
    y2 = next((y for y in range(rescaled_size-1, 0, -1) if min_vals[y] > 1), None) + 1

    return cv2.cvtColor(rendered_img[x1:x2, y1:y2], cv2.COLOR_BGR2RGB), imgray[x1:x2, y1:y2]>1

def cmp_box_intersection_w_areas(newbox:np.ndarray, oldboxes:np.ndarray, nboxes:int) -> bool:
    x21, y21, x22, y22 = np.split(oldboxes[:nboxes-1], 4, axis=1)

    xA = np.maximum(newbox[0], np.transpose(x21))
    yA = np.maximum(newbox[1], np.transpose(y21))
    xB = np.minimum(newbox[2], np.transpose(x22))
    yB = np.minimum(newbox[3], np.transpose(y22))

    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    boxAArea = (newbox[2] - newbox[0] + 1) * (newbox[3] - newbox[1] + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    if np.any(interArea/boxAArea > max_area_overlap):
        return False
    elif np.any(interArea/boxBArea > max_area_overlap):
        return False
    
    return True

def add_single_element(img:np.ndarray, obj_class:int, previous_elems:np.ndarray, iternum:int, box_size:int) -> bool:
    obj, mask = create_obj(obj_class, box_size)

    rnd_gen_offsets = np.array([img_width, img_height], dtype=np.float32) - (box_size+1)
    
    # decide position for the new object
    fails = 0
    while True:
        newbox = np.random.uniform(size=2).astype(np.float32) * rnd_gen_offsets
        newbox = np.hstack([newbox, newbox]).round().astype(np.int32)
        newbox[2] += mask.shape[1]
        newbox[3] += mask.shape[0]
        if cmp_box_intersection_w_areas(newbox, previous_elems[:-1,1:], iternum) or iternum == 0:
            break
        fails += 1
        if fails >= max_fails:
            return False # signal to restart
    
    # put object into image and newbox into previous_elems array
    x1,y1,x2,y2 = newbox.astype(int)
    img_zone = img[y1:y2, x1:x2]
    img_zone[mask,:] = obj[mask,:]
    previous_elems[iternum, 1:] = newbox
    previous_elems[iternum, 0] = obj_class

    return True


def add_armies(img:np.ndarray, boxes_coords:np.ndarray, ntanks:int, nflags:int, obj_size_px:int) -> bool:
    box_counter = 0

    for i in range(ntanks):
         if not add_single_element(img, np.random.randint(6), boxes_coords, box_counter, obj_size_px): return False
         box_counter += 1
    
    for i in range(nflags):
        if not add_single_element(img, np.random.randint(6,12), boxes_coords, box_counter, obj_size_px): return False
        box_counter += 1
    
    return True

def print_boxes(img:np.ndarray, boxes:np.ndarray):
    for i in range(boxes.shape[0]):
        color_id, x1, y1, x2, y2 = boxes[i]
        color_id = color_id % 6
        img[y1, x1:x2] = base_rgb_colors[color_id]
        img[y2, x1:x2] = base_rgb_colors[color_id]
        img[y1:y2, x1] = base_rgb_colors[color_id]
        img[y1:y2, x2] = base_rgb_colors[color_id]


def generate_dataset(n:int, backgrounds:list[str], dst_path:str, max_tanks:int=80, max_flags:int=20):
    out_imgs_dir, out_lbls_dir = os.path.join(dst_path, "images"), os.path.join(dst_path, "labels")
    scale_arr = np.array([img_width, img_height, img_width, img_height], dtype=np.float32)

    global render
    render = create_renderer()

    pbar = tqdm(total=n)

    i = len(os.listdir(out_imgs_dir))
    n += i
    while i < n:
        img = cv2.resize(cv2.imread(backgrounds[np.random.randint(len(backgrounds))], cv2.IMREAD_COLOR), [img_width, img_height])

        # to avoid segmentation fault of open3d due to bug -> recreate render and reload open3d workaround
        # if add_geometry_count // 65000 >= 1:
        #     del render
        #     global meshes
        #     del meshes
        #     reload(o3d)

        #     render = create_renderer()
        
        # need to decide how many tanks and flags to add based on the size of the objects themself or viceversa
        ntanks, nflags = np.random.randint(max_tanks), np.random.randint(max_flags)
        obj_size = np.random.randint(min_obj_size, max_obj_size)

        boxes = np.zeros([ntanks + nflags, 5], dtype=np.int32)
        if not add_armies(img, boxes, ntanks, nflags, obj_size):
            continue

        # save img and labels
        filenum_str = str(i).zfill(8)
        cv2.imwrite(os.path.join(out_imgs_dir, filenum_str + ".jpg"), img)

        boxes = boxes.astype(np.float32)
        boxes[:, 3:] -= boxes[:, 1:3]
        boxes[:, 1:3] += (boxes[:, 3:] / 2)
        boxes[:, 1:] /= scale_arr
        with open(os.path.join(out_lbls_dir, filenum_str + ".txt"), "w") as f:
            for j in range(boxes.shape[0]):
                f.write(str(int(boxes[j,0])) + " " + str(boxes[j,1]) + " " + str(boxes[j,2]) + " " + str(boxes[j,3]) + " " + str(boxes[j,4]) + "\n")

        pbar.update(1)
        i += 1
    pbar.close()

def main():
    parser = argparse.ArgumentParser(prog='dataset_generator', epilog='Risko dataset generator', description='Generate a synthetic dataset for the risiko problem')
    parser.add_argument('-o', '--output_dir', metavar='OUTPUT_DIR', type=str, default="generated_dataset", help='Directory in which the dataset will be generated')
    parser.add_argument('-b', '--backgrounds', metavar='BACKGROUNDS_DIR', type=str, default="backgrounds", help='Directory containing some images to use as backgrounds')
    parser.add_argument('-n', '--nimgs', metavar='GENERATED_IMAGES_NUMBER', type=int, default=1000, help='number of images to generate')
    parser.add_argument('-t', '--ntanks', metavar='N_TANKS', type=int, default=80, help='Max number of tanks generated in each image')
    parser.add_argument('-f', '--nflags', metavar='N_FLAGS', type=int, default=20, help='Max number of flags generated in each image')

    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        print("Output directory does not exists or it's not a directory")
        exit()
    if not os.path.isdir(os.path.join(output_dir, "images")):
        os.mkdir(os.path.join(output_dir, "images"))
    if not os.path.isdir(os.path.join(output_dir, "labels")):
        os.mkdir(os.path.join(output_dir, "labels"))

    background_dir = args.backgrounds
    if not os.path.exists(background_dir) or not os.path.isdir(background_dir):
        print("Backgrounds directory does not exists or it's not a directory")
        exit()
    
    n, ntanks, nflags = args.nimgs, args.ntanks, args.nflags

    backgrounds = [os.path.join(dp, f) for dp, dn, filenames in os.walk(background_dir) for f in filenames if os.path.splitext(f)[1] in [".jpg", ".JPG"]]

    np.random.seed(int(time.time()))

    generate_dataset(n, backgrounds, output_dir, ntanks, nflags)
    

if __name__ == "__main__":
    main()
