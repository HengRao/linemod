import pyrender
# import open3d as o3d
import numpy as np
import trimesh
# import matplotlib.pyplot as plt
import cv2
import time

__camera_look_at = np.array([0, 0, 0])
__intrinsic_matrix = np.array([
    [1150, 0, 640],
    [0, 1150, 480],
    [0, 0, 1]
])


def getCameraPose(lookAt, direction, distance):
    cameraZ = direction / np.linalg.norm(direction)
    cameraPosition = lookAt + distance * cameraZ
    # print("cameraPosition\n", cameraPosition)
    if abs(cameraZ[2]) < 0.999:
        cameraX = np.cross(np.array([0, 0, 1]), cameraZ)
        cameraX = cameraX / np.linalg.norm(cameraX)
        cameraY = np.cross(cameraZ, cameraX)
    else:
        cameraX = np.array([1, 0, 0])
        cameraY = np.cross(cameraZ, cameraX)

    cameraPose = np.eye(4)
    cameraPose[:3, 0] = cameraX
    cameraPose[:3, 1] = cameraY
    cameraPose[:3, 2] = cameraZ
    cameraPose[:3, 3] = cameraPosition
    return cameraPose


def getDirection(thetaXY, phiZ):
    z = np.sin(phiZ)
    x, y = np.cos(thetaXY), np.sin(thetaXY)
    xyNorm = np.abs(np.cos(phiZ))   # np.sqrt(1 - z * z)
    x = x * xyNorm
    y = y * xyNorm
    return np.array([x, y, z])


def __fix_camera_pose_generate(theta_xy, phi_z, distance):
    direction = getDirection(theta_xy, phi_z)
    camera_pose = getCameraPose(
        lookAt=__camera_look_at,
        direction=direction,
        distance=distance
    )
    return camera_pose


def transfer_model_to_templates(model_path, intrinsic_matrix, horizontal_angle_threshold, vertical_angle_threshold, distance):

    templates = []
    model_trimesh = trimesh.load(model_path)
    model_mesh = pyrender.Mesh.from_trimesh(model_trimesh)

    camera = pyrender.IntrinsicsCamera(intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])

    horizontal_times = int(360/horizontal_angle_threshold)
    vertical_times = int(360/vertical_angle_threshold)
    model_pose = np.eye(4)

    # rotate_times = int(360/rotate_angle)

    for i in range(horizontal_times):
        phi_z = i * horizontal_angle_threshold
        if (phi_z > 90 and phi_z <= 180) or (phi_z > 270):
            continue
        for j in range(vertical_times):
            theta_xy = j * vertical_angle_threshold
            if (phi_z == 90 or phi_z == 270) and theta_xy > 0:
                continue
            # print("phi_z: "+str(phi_z) + " theta_xy: "+str(theta_xy))
            camera_pose = __fix_camera_pose_generate(theta_xy*np.pi/180, phi_z*np.pi/180, distance)
            scene = pyrender.Scene(ambient_light=np.array([1, 1, 1, 1.0]))
            model_node = scene.add(model_mesh, pose=model_pose)
            camera_node = scene.add(camera, pose=camera_pose)

            r = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=960)
            color, depth = r.render(scene)

            templates.append(color)
            templates.append(depth*1000)

            # cv2.imshow("depth", color)
            # cv2.waitKey(1)

            r.delete()

    return templates


def rotate_templates_and_masks(templates, masks, angle):

    rotate_templates = []
    rotate_masks = []
    rotate_times = int(360/angle)
    M = np.zeros((2, 3), dtype=np.float32)

    for t in range(int(len(templates)/2)):
        color = templates[2*t]
        depth = templates[2*t+1]
        mask = masks[t]

        for x in range(rotate_times):
            # print("process {} start.".format(x))
            rotate_angle = x * angle

            center = (int(color.shape[1]/2), int(color.shape[0]/2))

            M = cv2.getRotationMatrix2D(center, rotate_angle, 1)

            rotate_color = cv2.warpAffine(color, M, (int(color.shape[1]), int(color.shape[0])), borderValue=(255, 255, 255))
            rotate_depth = cv2.warpAffine(depth, M, (int(depth.shape[1]), int(depth.shape[0])), borderValue=0)
            rotate_mask = cv2.warpAffine(mask, M, (int(mask.shape[1]), int(mask.shape[0])), borderValue=0)

            rotate_templates.append(rotate_color)
            rotate_templates.append(rotate_depth)
            rotate_masks.append(rotate_mask)

            # print("process {} end.".format(x))
            cv2.imshow("color", rotate_color)
            cv2.waitKey(2)

    return rotate_templates, rotate_masks


def get_mask_from_templates(templates):
    masks = []
    for i in range(int(len(templates)/2)):
        tmp = templates[2*i+1]
        mk = np.zeros(tmp.shape, dtype=np.uint8)
        mk[tmp > 0] = 255

        # cv2.imshow("mask", mk)
        # cv2.waitKey(0)

        masks.append(mk)

    print("length:" + str(len(masks)))

    return masks


model_path = 'pyrender/examples/models/fuze.obj'
theta_xy = 45
phi_z = 45
distance = 0.5
rotate_angle = 20

time_start = time.time()
templates = transfer_model_to_templates(model_path, __intrinsic_matrix, phi_z, theta_xy, distance)
# print(len(templates))
time_end = time.time()
print("render time: ", time_end-time_start)

time_start = time.time()
masks = get_mask_from_templates(templates)
time_end = time.time()
print("generate masks time: ", time_end-time_start)

time_start = time.time()
rotate_tmp, rotate_mks = rotate_templates_and_masks(templates, masks, rotate_angle)
time_end = time.time()
print("rotate template time: ", time_end-time_start)

color_path = 'pyrender/examples/color.png'
depth_path = 'pyrender/examples/depth.png'

color_img = cv2.imread(color_path)
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
depth_img = cv2.imread(depth_path)
print(depth_img.dtype)
print(np.unique(depth_img))

rgb = cv2.imread('sphere/test/000000/rgb/000003.jpg')
print(type(rgb))
print(rgb.dtype)
print(type(templates[1]))

# cv2.imshow(color_img)
# cv2.imshow(depth_img)

colorGradient = cv2.linemod.ColorGradient_create(10, 63, 55)
depthNormal = cv2.linemod.DepthNormal_create(2000, 20, 20, 4)
T = [5, 8]
modalities = []
modalities.append(colorGradient)
# modalities.append(depthNormal)
detector = cv2.linemod_Detector(modalities, T)

bounding_boxes = []

# cv2.imshow('mask1', masks[0])
# print(len(masks))

for i in range(len(rotate_mks)):
    src = []
    src.append(rotate_tmp[2*i])
    # print(templates[2*i+1][300, 680])
    # src.append(templates[2*i+1])
    ret, boundingbox = detector.addTemplate(src, '1', rotate_mks[i])
    cv2.rectangle(rotate_mks[i], (boundingbox[0], boundingbox[1]), (boundingbox[0]+boundingbox[2], boundingbox[1]+boundingbox[3]), 255, 2)
    # cv2.imshow('depth', masks[i])
    bounding_boxes.append(boundingbox)

    # cv2.waitKey(1)
    # print(ret)


dets = []
dets.append(color_img)

matches, quantized_images = detector.match(dets, 85, '1')

for n in matches:
    template_id = n.template_id
    template = detector.getTemplates('1', template_id)

    cv2.rectangle(color_img, (n.x, n.y), (n.x+template[0].width, n.y+template[0].height), (0, 255, 0), 2)

    print(template[0].width)
    print(template[0].height)


for m in matches:
    print(m.x, m.y)
    print(m.similarity)
    print(m.template_id)
    cv2.circle(color_img, (m.x, m.y), 1, (255, 0, 0), 1)
    # print(str(i.x) + " " + str(i.y))

cv2.imshow('rgb', color_img)

cv2.waitKey(0)
