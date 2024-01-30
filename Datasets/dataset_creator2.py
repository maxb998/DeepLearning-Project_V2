import os, cv2, colorsys, copy, argparse, time
import numpy as np
import open3d as o3d
from tqdm import tqdm

class Params:
    outDir:str
    backgroundsDir:str
    nimgs:int
    maxtanks:int
    maxflags:int
    imgWidth:int
    imgHeight:int
    maxObjSize:int
    minObjSize:int
    renderRes:int
    maxOverlap:float
    renderCount:int = int(0)
    

# CONSTANTS
meshes = {
    'tank':o3d.io.read_triangle_mesh('Datasets/3D_models/Risiko_Tank_v3.stl'),
    'flag':o3d.io.read_triangle_mesh('Datasets/3D_models/RisikoFlag_V2.stl')
}
camera_center = np.zeros(3, dtype=np.float32)
erosion_kernel = np.ones(shape=(7,7), dtype=np.uint8) #np.array(((0,1,0),(1,1,1),(0,1,0)), dtype=np.uint8)
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



def restricted_int(x):
    x = int(x)
    if x < 0:
        raise argparse.ArgumentTypeError("Cannot be less than 0")
    return x
    

def argParser() -> Params:
    parser = argparse.ArgumentParser(prog='dataset_generator', epilog='Risko dataset generator', description='Generate a synthetic dataset for the risiko problem')
    parser.add_argument('-O', '--outputDir', metavar='OUTPUT_DIR', type=str, required=True, help='Directory in which the dataset will be generated')
    parser.add_argument('-B', '--backgroundsDir', metavar='BACKGROUNDS_DIR', type=str, required=True, help='Directory containing some images to use as backgrounds')
    parser.add_argument('-N', '--nimgs', metavar='GENERATED_IMAGES_NUMBER', type=restricted_int, required=True, help='number of images to generate')
    parser.add_argument('--maxtanks', metavar='N_TANKS', type=restricted_int, default=80, help='Max number of tanks generated in each image')
    parser.add_argument('--maxflags', metavar='N_FLAGS', type=restricted_int, default=20, help='Max number of flags generated in each image')
    parser.add_argument('--imgWidth', metavar='IMG_WIDTH', type=restricted_int, default=1000, help='Output images width in pixels')
    parser.add_argument('--imgHeight', metavar='IMG_HEIGHT', type=restricted_int, default=1000, help='Output images height in pixels')
    parser.add_argument('--maxObjSize', metavar='MAX_OBJ_SIZE', type=restricted_int, default=250, help='Biggest tank and flag resize after render in pixels(not exactly the max size)')
    parser.add_argument('--minObjSize', metavar='MIN_OBJ_SIZE', type=restricted_int, default=20, help='Smallest tank and flag resize after render in pixels(not actually the min size)')
    parser.add_argument('--renderRes', metavar='RENDER_RES', type=restricted_int, help='Output images height in pixels (default is twice maxObjSize). If set lower than maxObjSize than rendered objects might be upscaled')
    parser.add_argument('--maxOverlap', metavar='MAX_OVERLAP', type=float, default=0.6, help='The maximum amount an object can be overlapped by another. Value range: (0,1)')

    args = parser.parse_args()

    p = Params()

    p.outDir = args.outputDir
    if not os.path.exists(p.outDir) or not os.path.isdir(p.outDir):
        raise Exception("Output directory does not exists or it's not a directory")
    
    # check if there and generate "images" and "labels" directories
    if not os.path.isdir(os.path.join(p.outDir, "images")):
        os.mkdir(os.path.join(p.outDir, "images"))
    if not os.path.isdir(os.path.join(p.outDir, "labels")):
        os.mkdir(os.path.join(p.outDir, "labels"))

    p.backgroundsDir = args.backgroundsDir
    if not os.path.exists(p.backgroundsDir) or not os.path.isdir(p.backgroundsDir):
        raise Exception("Backgrounds directory does not exists or it's not a directory")
    
    p.nimgs = args.nimgs
    p.maxtanks = args.maxtanks
    p.maxflags = args.maxflags
    p.imgWidth = args.imgWidth
    p.imgHeight = args.imgHeight
    p.maxObjSize = args.maxObjSize
    p.minObjSize = args.minObjSize

    temp = args.renderRes
    if temp == None:
        temp = p.maxObjSize * int(2)
    p.renderRes = temp

    p.maxOverlap = args.maxOverlap

    return p

def create_renderer(p:Params) -> o3d.visualization.rendering.OffscreenRenderer:

    renderer = o3d.visualization.rendering.OffscreenRenderer(p.renderRes, p.renderRes)
    renderer.scene.scene.enable_sun_light(False)
    renderer.scene.set_lighting(renderer.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
    renderer.scene.set_background([0,0,0,0])

    v_fov      = 15.0       # vertical field of view: between 5 and 90 degrees
    near_plane = 0.1
    far_plane  = 1e3
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    renderer.scene.camera.set_projection(v_fov, 1, near_plane, far_plane, fov_type)

    renderer.scene.add_geometry('tank', copy.deepcopy(meshes['tank']), create_material_record(1))
    renderer.scene.add_geometry('flag', copy.deepcopy(meshes['flag']), create_material_record(1))

    return renderer

def create_material_record(obj_class:int) -> o3d.visualization.rendering.MaterialRecord:

    color = np.random.uniform(size=3).astype(np.float64)
    color = color * hsv_scalers[obj_class % 6] + hsv_offsets[obj_class % 6]
    rgba_color = np.ones(4, dtype=np.float64)
    rgba_color[:3] = colorsys.hsv_to_rgb(color[0], color[1], color[2])

    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = rgba_color  # RGBA
    mtl.shader = "defaultLit"

    return mtl

def modify_obj(renderer:o3d.visualization.rendering.OffscreenRenderer, obj_class:int):

    key = 'tank'
    if obj_class / 6 >= 1:
        key = 'flag'

    # change obj color
    material = create_material_record(obj_class)
    renderer.scene.modify_geometry_material(key, material)

    # rotate obj
    rotation_mat = np.eye(N=4, dtype=np.float64)
    rotation_mat[0:3,0:3] = copy.deepcopy(meshes[key]).get_rotation_matrix_from_xyz((np.random.rand(3) * 10))
    renderer.scene.set_geometry_transform(key, rotation_mat)

    # rotate camera around to have different lightning orientations
    camera_eye = np.random.rand(3)
    camera_eye /= np.linalg.norm(camera_eye)
    camera_eye *= 90
    camera_up = np.random.rand(3)
    camera_up /= np.linalg.norm(camera_up)
    renderer.scene.camera.look_at(camera_center, camera_eye, camera_up)


def render_obj(renderer:o3d.visualization.rendering.OffscreenRenderer, rescaled_size:int) -> np.ndarray:

    # Read the image into a variable
    rendered_img = np.array(renderer.render_to_image())

    rendered_img = rendered_img.astype(np.uint8)
    imgray = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2GRAY)
    imgray[imgray > 1] = 255
    cv2.erode(imgray, erosion_kernel, imgray, iterations=1)
    rendered_img = cv2.GaussianBlur(rendered_img, (13,13), 9)
    rescaled_img = cv2.resize(rendered_img, [rescaled_size, rescaled_size], interpolation=cv2.INTER_LINEAR)
    rescaled_imgray = cv2.resize(imgray, [rescaled_size, rescaled_size], interpolation=cv2.INTER_LINEAR)
    rescaled_imgray[rescaled_imgray < 255] = 0
    
    min_vals = np.max(rescaled_imgray, axis=1)
    x1 = next((x for x in range(rescaled_size) if min_vals[x] > 1), None)
    x2 = next((x for x in range(rescaled_size-1, 0, -1) if min_vals[x] > 1), None) + 1
    np.max(rescaled_imgray, axis=0, out=min_vals)
    y1 = next((y for y in range(rescaled_size) if min_vals[y] > 1), None)
    y2 = next((y for y in range(rescaled_size-1, 0, -1) if min_vals[y] > 1), None) + 1

    rescaled_img = cv2.cvtColor(rescaled_img[x1:x2, y1:y2], cv2.COLOR_BGR2RGB)
    mask = rescaled_imgray[x1:x2, y1:y2]>1

    rescaled_img[mask==False] = (255,255,255)

    return rescaled_img, mask

def cmp_box_intersection_w_areas(newbox:np.ndarray, oldboxes:np.ndarray, nboxes:int, maxAreaOverlap:float) -> tuple[np.ndarray, np.ndarray]:
    x21, y21, x22, y22 = np.split(oldboxes[:nboxes-1], 4, axis=1)

    xA = np.maximum(newbox[0], np.transpose(x21))
    yA = np.maximum(newbox[1], np.transpose(y21))
    xB = np.minimum(newbox[2], np.transpose(x22))
    yB = np.minimum(newbox[3], np.transpose(y22))

    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    boxAArea = (newbox[2] - newbox[0] + 1) * (newbox[3] - newbox[1] + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    if np.any(interArea/boxAArea > maxAreaOverlap):
        return False
    elif np.any(interArea/boxBArea > maxAreaOverlap):
        return False
    
    return True

def add_single_element(p:Params, renderer:o3d.visualization.rendering.OffscreenRenderer, img:np.ndarray, obj_class:int, previous_elems:np.ndarray, iternum:int, box_size:int) -> bool:

    modify_obj(renderer, obj_class)
    obj, mask = render_obj(renderer, box_size)
    p.renderCount += 1

    rnd_gen_offsets = np.array([p.imgWidth, p.imgHeight], dtype=np.float32) - (box_size+1)
    
    # decide position for the new object
    fails = 0
    while True:
        newbox = np.random.uniform(size=2).astype(np.float32) * rnd_gen_offsets
        newbox = np.hstack([newbox, newbox]).round().astype(np.int32)
        newbox[2] += mask.shape[1]
        newbox[3] += mask.shape[0]
        if cmp_box_intersection_w_areas(newbox, previous_elems[:-1,1:], iternum, p.maxOverlap) or iternum == 0:
            break
        fails += 1
        if fails >= 20:
            return False # signal to restart
    
    # put object into image and newbox into previous_elems array
    x1,y1,x2,y2 = newbox.astype(int)
    img_zone = img[y1:y2, x1:x2]
    img_zone[mask,:] = obj[mask,:]
    previous_elems[iternum, 1:] = newbox
    previous_elems[iternum, 0] = obj_class

    return True


def add_armies(p:Params, renderer:o3d.visualization.rendering.OffscreenRenderer, img:np.ndarray, boxes_coords:np.ndarray, ntanks:int, nflags:int, obj_size_px:int) -> bool:
    box_counter = 0

    renderer.scene.show_geometry('tank', True)
    renderer.scene.show_geometry('flag', False)
    for i in range(ntanks):
         if not add_single_element(p, renderer, img, np.random.randint(6), boxes_coords, box_counter, obj_size_px): return False
         box_counter += 1
    
    renderer.scene.show_geometry('tank', False)
    renderer.scene.show_geometry('flag', True)
    for i in range(nflags):
        if not add_single_element(p, renderer, img, np.random.randint(6,12), boxes_coords, box_counter, obj_size_px): return False
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


def generate_dataset(p:Params):
    out_imgs_dir, out_lbls_dir = os.path.join(p.outDir, "images"), os.path.join(p.outDir, "labels")
    scale_arr = np.array([p.imgWidth, p.imgHeight, p.imgWidth, p.imgHeight], dtype=np.float32)

    backgrounds = [os.path.join(dp, f) for dp, dn, filenames in os.walk(p.backgroundsDir) for f in filenames if os.path.splitext(f)[1] in [".jpg", ".JPG"]]

    renderer = create_renderer(p)

    #pbar = tqdm(total=p.nimgs)

    ntanks, nflags = 0, 0

    i = len(os.listdir(out_imgs_dir))
    n = p.nimgs + i
    while i < n:
        img = cv2.resize(cv2.imread(backgrounds[np.random.randint(len(backgrounds))], cv2.IMREAD_COLOR), [p.imgWidth, p.imgHeight])
        
        # need to decide how many tanks and flags to add based on the size of the objects themself or viceversa
        if p.maxtanks > 0:
            ntanks = np.random.randint(p.maxtanks)
        if p.maxflags > 0:
            np.random.randint(p.maxflags)

        obj_size = np.random.randint(p.minObjSize, p.maxObjSize)

        boxes = np.zeros([ntanks + nflags, 5], dtype=np.int32)
        if not add_armies(p, renderer, img, boxes, ntanks, nflags, obj_size):
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

        print('image num = ' + str(i - (n - p.nimgs)) + '\t  renderCount = ' + str(p.renderCount))
        #pbar.update(1)
        i += 1
    #pbar.close()


def main():
    p = argParser()

    np.random.seed(int(time.time()))

    generate_dataset(p)

if __name__ == "__main__":
    main()
