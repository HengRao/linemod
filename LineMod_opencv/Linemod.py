import pyrender
# import open3d as o3d
import numpy as np
import trimesh
# import matplotlib.pyplot as plt
import cv2
import time
from Framework.PoseEstimation import ClassicDetector
from Framework.PoseEstimation import classicDetectorRegistry

default_profile = {
    'color_gradient_weak_threshold': 5,
    'color_gradient_features': 63,
    'color_gradient_strong_threshold': 55,
    'depth_normal_distance_threshold': 2000,
    'depth_normal_difference_thgreshold': 50,
    'depth_normal_num_features': 63,
    'depth_normal_extract_threshold': 2

}

@classicDetectorRegistry.register()
class LinemodOpencv(ClassicDetector):
    def __init__(self, color_gradient_weak_threshold = 5, color_gradient_features = 63, color_gradient_strong_threshold = 55, depth_normal_distance_threshold = 2000, depth_normal_difference_thgreshold = 50, depth_normal_num_features = 63, depth_normal_extract_threshold = 2, T_pyramid = [5, 8]):
        super(LinemodOpencv, self).__init__()
        self._colorGradient = cv2.linemod.ColorGradient_create(color_gradient_weak_threshold, color_gradient_features, color_gradient_strong_threshold)
        self._depthNormal = cv2.linemod.DepthNormal_create(depth_normal_distance_threshold, depth_normal_difference_thgreshold, depth_normal_num_features, depth_normal_extract_threshold)
        self.T_pyramid = T_pyramid
        self.modalities = []
        self.modalities.append(colorGradient)
        # modalities.append(depthNormal)
        # self._detector = cv2.linemod_Detector(modalities, T_pyramid)

    def register(self, model, camera_intrinsic_matrix, class_id = '1', horizontal_angle_threshold = 30, vertical_angle_threshold = 30, distance = 0.7, rotate_angle = 30):
        self.detector = cv2.linemod_Detector(self.modalities, self.T_pyramid)

        model_mesh = pyrender.Mesh.from_trimesh(model)
        camera = pyrender.IntrinsicsCamera(camera_intrinsic_matrix[0, 0], camera_intrinsic_matrix[1, 1], camera_intrinsic_matrix[0, 2], camera_intrinsic_matrix[1, 2])
        
        horizontal_times = int(360/horizontal_angle_threshold)
        vertical_times = int(360/vertical_angle_threshold)
        rotate_times = int(360/rotate_angle)

        templates_sum = 0

        for i in range(horizontal_times):
            phi_z = i * horizontal_angle_threshold
            if (phi_z > 90 and phi_z <= 180) or (phi_z > 270):
                continue
            for j in range(vertical_times):
                theta_xy = j * vertical_angle_threshold
                if (phi_z == 90 or phi_z == 270) and theta_xy > 0:
                    continue

                color, depth = render_model_with_angle(camera, model_mesh, theta_xy, phi_z, distance)

                for r in range(rotate_times):
                    r_angle = r * rotate_angle

                    rotate_color = rotate_template(color, rotate_angle, (255, 255, 255))
                    rotate_depth = rotate_template(depth, rotate_angle, 0)
                    rotate_mask = rotate_template(mask, rotate_angle, 0)

                    sources = []
                    sources.append(rotate_color)
                    sources.append(rotate_depth)

                    ret, boundingbox = self.detector.addTemplate(sources, class_id, rotate_mask)

                    if ret > -1:
                        templates_sum += 1


        return templates_sum



    def render_model_with_angle(camera, model_mesh, theta_xy, phi_z, distance):
        camera_pose = fix_camera_pose_generate(theta_xy*np.pi/180, phi_z*np.pi/180, distance)
        model_pose = np.eye(4)

        scene = pyrender.Scene(ambient_light=np.array([1, 1, 1, 1.0]))
        model_node = scene.add(model_mesh, pose=model_pose)
        camera_node = scene.add(camera, pose=camera_pose)

        r = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=960)
        color, depth = r.render(scene)

        return color, depth


    def rotate_template(template, rotate_angle, fill_color):

         center = (int(template.shape[1]/2), int(template.shape[0]/2))
         M = cv2.getRotationMatrix2D(center, rotate_angle, 1)
         rotate_template = cv2.warpAffine(template, M, (int(template.shape[1]), int(template.shape[0])), borderValue=fill_color)

         return rotate_template


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


    def fix_camera_pose_generate(theta_xy, phi_z, distance):
        direction = getDirection(theta_xy, phi_z)
        camera_pose = getCameraPose(
            lookAt=__camera_look_at,
            direction=direction,
            distance=distance
        )
        return camera_pose

