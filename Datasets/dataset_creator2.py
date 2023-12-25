
#from matplotlib import pyplot as plt
import os, cv2, colorsys, copy, argparse, time
import numpy as np
import open3d as o3d
from tqdm import tqdm


## General Params
img_width, img_height = 1000, 1000
min_obj_size, max_obj_size = 15, 250
max_fails = 20
max_area_overlap = 0.6

## Rendering Params
render_res = 400
center = [0, 0, 0]  # look_at target
rpy_amp    = 10         # rotation span (for randomly drawing)

models_path = [ 'Datasets/3D_models/Risiko_Tank_v3.stl', 'Datasets/3D_models/RisikoFlag_V2.stl' ]

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
], dtype=np.float64)
hsv_scalers = np.array([
    [ 9/360, 0.12,  0.5], # blue
    [21/360, 0.12,  0.5], # red
    [21/360,  0.3,  0.5], # yellow
    [22/360,  0.3,  0.5], # purple
    [    1,   0.1, 0.05], # black
    [61/360,  0.3,  0.5]  # green
], dtype=np.float64)

def create_renderer(mesh_path:str) -> tuple[o3d.visualization.rendering.OffscreenRenderer, o3d.geometry.TriangleMesh]:

    mesh = o3d.io.read_triangle_mesh(mesh_path)

    renderer = o3d.visualization.rendering.OffscreenRenderer(render_res, render_res)
    renderer.scene.scene.enable_sun_light(False)
    renderer.scene.set_lighting(renderer.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
    renderer.scene.set_background([0,0,0,0])

    v_fov      = 15.0       # vertical field of view: between 5 and 90 degrees
    near_plane = 0.1
    far_plane  = 1e3
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    renderer.scene.camera.set_projection(v_fov, 1, near_plane, far_plane, fov_type)
    eye = [0, 0, 90]  # camera position
    up = [0, 1, 0]  # camera orientation
    renderer.scene.camera.look_at(center, eye, up)

    mesh_copy = copy.deepcopy(mesh)
    renderer.scene.add_geometry("myobj", mesh_copy, create_material_record(0))

    return renderer, mesh

def create_material_record(obj_class:int) -> o3d.visualization.rendering.MaterialRecord:

    color = np.random.uniform(size=3).astype(np.float64)
    color = color * hsv_scalers[obj_class % 6] + hsv_offsets[obj_class % 6]
    rgba_color = np.ones(4, dtype=np.float64)
    rgba_color[:3] = colorsys.hsv_to_rgb(color[0], color[1], color[2])

    mtl = o3d.visualization.rendering.MaterialRecord()    # or MaterialRecord(), for later versions of Open3D
    mtl.base_color = rgba_color  # RGBA
    mtl.shader = "defaultLit"

    return mtl

def modify_obj(renderer:o3d.visualization.rendering.OffscreenRenderer, mesh:o3d.geometry.TriangleMesh, obj_class:int):

    # change obj color
    material = create_material_record(obj_class)
    renderer.scene.modify_geometry_material('myobj', material)

    # rotate obj
    rotation_mat = np.eye(N=4, dtype=np.float64)
    rotation_mat[0:3,0:3] = mesh.get_rotation_matrix_from_xyz((np.random.rand(3) * rpy_amp))
    renderer.scene.set_geometry_transform('myobj', rotation_mat)


def render_obj(renderer:o3d.visualization.rendering.OffscreenRenderer, rescaled_size:int) -> np.ndarray:

    # Read the image into a variable
    rendered_img = np.array(renderer.render_to_image())

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

def cmp_box_intersection_w_areas(newbox:np.ndarray, oldboxes:np.ndarray, nboxes:int) -> tuple[np.ndarray, np.ndarray]:
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

def add_single_element(renderer:o3d.visualization.rendering.OffscreenRenderer, mesh:o3d.geometry.TriangleMesh, img:np.ndarray, obj_class:int, previous_elems:np.ndarray, iternum:int, box_size:int) -> bool:

    modify_obj(renderer, mesh, obj_class)
    obj, mask = render_obj(renderer, box_size)

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


def add_armies(renderers:list, meshes:list, img:np.ndarray, boxes_coords:np.ndarray, ntanks:int, nflags:int, obj_size_px:int) -> bool:
    box_counter = 0

    for i in range(ntanks):
         if not add_single_element(renderers[0], meshes[0], img, np.random.randint(6), boxes_coords, box_counter, obj_size_px): return False
         box_counter += 1
    
    for i in range(nflags):
        if not add_single_element(renderers[1], meshes[1], img, np.random.randint(6,12), boxes_coords, box_counter, obj_size_px): return False
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

    renderes = []
    meshes = []
    for mesh_path in models_path:
        r, m = create_renderer(mesh_path)
        renderes.append(r)
        meshes.append(m)

    pbar = tqdm(total=n)

    i = len(os.listdir(out_imgs_dir))
    n += i
    while i < n:
        img = cv2.resize(cv2.imread(backgrounds[np.random.randint(len(backgrounds))], cv2.IMREAD_COLOR), [img_width, img_height])
        
        # need to decide how many tanks and flags to add based on the size of the objects themself or viceversa
        ntanks, nflags = np.random.randint(max_tanks), np.random.randint(max_flags)
        obj_size = np.random.randint(min_obj_size, max_obj_size)

        boxes = np.zeros([ntanks + nflags, 5], dtype=np.int32)
        if not add_armies(renderes, meshes,img, boxes, ntanks, nflags, obj_size):
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

    print('\nALL DONE! IGNORE THE ERROR\n')


def main():
    parser = argparse.ArgumentParser(prog='dataset_generator', epilog='Risko dataset generator', description='Generate a synthetic dataset for the risiko problem')
    parser.add_argument('-o', '--output_dir', metavar='OUTPUT_DIR', type=str, required=True, help='Directory in which the dataset will be generated')
    parser.add_argument('-b', '--backgrounds', metavar='BACKGROUNDS_DIR', type=str, required=True, help='Directory containing some images to use as backgrounds')
    parser.add_argument('-n', '--nimgs', metavar='GENERATED_IMAGES_NUMBER', type=int, required=True, help='number of images to generate')
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
