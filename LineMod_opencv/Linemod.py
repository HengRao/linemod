import pyrender
# import open3d as o3d
import numpy as np
import trimesh
# import tkinter
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import time
from DFAi3DCore.Framework.PoseEstimation.ClassicDetector import ClassicDetector
# from Framework.PoseEstimation import classicDetectorRegistry

default_profile = {
    'color_gradient_weak_threshold': 5,
    'color_gradient_features': 63,
    'color_gradient_strong_threshold': 55,
    'depth_normal_distance_threshold': 2000,
    'depth_normal_difference_thgreshold': 50,
    'depth_normal_num_features': 63,
    'depth_normal_extract_threshold': 2
}


# @classicDetectorRegistry.register()
class LinemodOpencv(ClassicDetector):

    def __init__(self, color_gradient_weak_threshold = 5, color_gradient_features = 63, color_gradient_strong_threshold = 55, \
                 depth_normal_distance_threshold = 5000, depth_normal_difference_thgreshold = 100, depth_normal_num_features = 63, \
                 depth_normal_extract_threshold = 2, T_pyramid = [5, 8] ):
        # super(LinemodOpencv, self).__init__()
        self.colorGradient = cv2.linemod.ColorGradient_create(color_gradient_weak_threshold, color_gradient_features, color_gradient_strong_threshold)
        self.depthNormal = cv2.linemod.DepthNormal_create(depth_normal_distance_threshold, depth_normal_difference_thgreshold, depth_normal_num_features, depth_normal_extract_threshold)
        self.T_pyramid = T_pyramid
        self.modalities = []
        self.modalities.append(self.colorGradient)
        # self.modalities.append(self.depthNormal)


    def register(self, model, camera_intrinsic_matrix, class_id = '1', horizontal_angle_threshold = 60, vertical_angle_threshold = 60, \
        distance = 2, rotate_angle = 30):
        self.detector = cv2.linemod_Detector(self.modalities, self.T_pyramid)
        self.class_id = class_id
        self.camera_intrinsic_matrix = camera_intrinsic_matrix

        model_mesh = pyrender.Mesh.from_trimesh(model)

        horizontal_times = int(360/horizontal_angle_threshold)
        vertical_times = int(360/vertical_angle_threshold)
        rotate_times = int(360/rotate_angle)

        templates_sum = 0

        # plt.figure()
        for d in range(30):
            for i in range(horizontal_times):
                phi_z = i * horizontal_angle_threshold
                if (phi_z > 90 and phi_z <= 180) or (phi_z > 270):
                    continue
                for j in range(vertical_times):
                    theta_xy = j * vertical_angle_threshold
                    if (phi_z == 90 or phi_z == 270) and theta_xy > 0:
                        continue

                    color, depth = self.render_model_with_angle(model_mesh, theta_xy, phi_z, distance - d*10)
                    mask = self.get_mask_from_templates(depth)

                    depth = depth.astype('short')
                    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite("color.jpg", color)
                    # cv2.imwrite("depth.jpg", depth)
                    # print(np.unique(depth))  
                    # print(color.shape)

                    for r in range(rotate_times):
                        r_angle = r * rotate_angle

                        rotate_color = self.rotate_template(color, r_angle, (255, 255, 255))
                        rotate_depth = self.rotate_template(depth, r_angle, 0)
                        rotate_mask = self.rotate_template(mask, r_angle, 0)
  
                        rotate_depth = rotate_depth.astype('short')

                        sources = []
                        sources.append(rotate_color)
                        # sources.append(rotate_depth)

                        cv2.imwrite("color.jpg", rotate_color)

                        ret, boundingbox = self.detector.addTemplate(sources, class_id, rotate_mask)

                        if ret > -1:
                            templates_sum += 1

        return templates_sum


    def register_with_rgbd(self, images, class_id = '1'):
        mask = self.get_mask_from_templates(images[0])
        ret, boundingbox = self.detector.addTemplate(images,  class_id, mask)
        return ret


    def detect(selfc, images,  similarity_score):
        mt = []
        mt.append(images[0])
        matches, quantized_images = self.detector.match(mt, similarity_score, self.class_id)

        # for m in matches:
        if len(matches)>0:
            boxes = []
            scores = []
            for m in matches:

                x = m.x
                y = m.y

                template_id = m.template_id
                template = self.detector.getTemplates(self.class_id, template_id)

                width = template[0].width
                height = template[0].height

                r = [x, y, x+width, y+height]

                s = m.similarity

                boxes.append(r)
                scores.append(s)

            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)

            poses = []

            for i in indices:
                # print(i[0])
                index = i[0]
                cv2.rectangle(images[0], (boxes[index][0], boxes[index][1]), (boxes[index][2], boxes[index][3]), (0, 255, 0), 2)
                # cv2.circle(images[0],  )
                cv2.imwrite("result.jpg", images[0])

                x = boxes[index][0] + (boxes[index][2] - boxes[index][0])/2
                y = boxes[index][1] + (boxes[index][3] - boxes[index][1])/2
                z = images[1][int(x), int(y)]
                # print(z)

                pose = self.get_object_camera_pose(x, y, z)

                poses.append(pose)

            return poses

        else:
            return None


    def get_object_camera_pose(self, u, v, z):
        uv = np.array([u, v, 1])
        # print(uv)
        k_inv = np.linalg.inv(self.camera_intrinsic_matrix)
        # print(k_inv)

        camera_pose = np.matmul(k_inv, z*uv)

        return camera_pose


    def get_mask_from_templates(self, template):

        mask = np.zeros(template.shape, dtype=np.uint8)
        mask[template > 0] = 255

        return mask


    def render_model_with_angle(self, model_mesh, theta_xy, phi_z, distance):
        camera_pose = self.fix_camera_pose_generate(theta_xy*np.pi/180, phi_z*np.pi/180, distance)
        # print(camera_pose)
        model_pose = np.eye(4)
        fx = self.camera_intrinsic_matrix[0, 0]
        fy = self.camera_intrinsic_matrix[1, 1]
        cx = self.camera_intrinsic_matrix[0, 2]
        cy = self.camera_intrinsic_matrix[1, 2]
        camera = pyrender.IntrinsicsCamera(fx, fy, \
                                cx, cy, zfar=2*distance)

        camera_p = pyrender.PerspectiveCamera(yfov=np.pi/3, aspectRatio=1.0)

        scene = pyrender.Scene(ambient_light=np.array([1, 1, 1, 1.0]))
        model_node = scene.add(model_mesh, pose=model_pose)
        camera_node = scene.add(camera, pose=camera_pose)

        r = pyrender.OffscreenRenderer(viewport_width=2*cx, viewport_height=2*cy)
        color, depth = r.render(scene)

        r.delete()
        # print(np.unique(depth))  

        if distance > 10:
            return color, depth
        else:
            return color, depth


    def rotate_template(self, template, rotate_angle, fill_color):

            center = (int(template.shape[1]/2), int(template.shape[0]/2))
            M = cv2.getRotationMatrix2D(center, rotate_angle, 1)
            rotate_template = cv2.warpAffine(template, M, (int(template.shape[1]), int(template.shape[0])), borderValue=fill_color)

            return rotate_template


    def getCameraPose(self, lookAt, direction, distance):
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


    def getDirection(self, thetaXY, phiZ):
        z = np.sin(phiZ)
        x, y = np.cos(thetaXY), np.sin(thetaXY)
        xyNorm = np.abs(np.cos(phiZ))   # np.sqrt(1 - z * z)
        x = x * xyNorm
        y = y * xyNorm
        return np.array([x, y, z])


    def fix_camera_pose_generate(self, theta_xy, phi_z, distance):
        direction = self.getDirection(theta_xy, phi_z)
        camera_pose = self.getCameraPose(
            lookAt = np.array([0, 0, 0]),
            direction=direction,
            distance=distance
        )
        return camera_pose

