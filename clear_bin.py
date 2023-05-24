import sim
import os
import camera
import pybullet as p
import numpy as np
import torch
import train_seg_model
import torchvision
import icp
import transforms
from torchvision import transforms as t
from scipy.spatial.transform import Rotation
import random
from rrt import *
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import train_seg_model 


mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]

class ImgDataset(Dataset):
    def __init__(self, img):
        """
            Initialize instance variables.
            :param img_dir (str): path of train or test folder.
            :return None:
        """
        # TODO: complete this method
        # ===============================================================================
        self.transform = t.Compose([t.ToTensor(),t.Normalize(mean_rgb, std_rgb)])
        self.dataset_length = 1 
        self.img=img
        # ===============================================================================

    def __len__(self):
        """
            Return the length of the dataset.
            :return dataset_length (int): length of the dataset, i.e. number of samples in the dataset
        """
        # TODO: complete this method
        # ===============================================================================
        return self.dataset_length
        # ===============================================================================

    def __getitem__(self, idx):
        """
            Given an index, return paired rgb image and ground truth mask as a sample.
            :param idx (int): index of each sample, in range(0, dataset_length)
            :return sample: a dictionary that stores paired rgb image and corresponding ground truth mask.
        """
        # TODO: complete this method
        # Hint:
        # - Use image.read_rgb() and image.read_mask() to read the images.
        # - Think about how to associate idx with the file name of images.
        # - Remember to apply transform on the sample.
        # ===============================================================================
        rgb_img=self.transform(self.img)
        sample = {'input': rgb_img}
        return sample
        # ===============================================================================


if __name__ == "__main__":
    if not os.path.exists('checkpoint.pth.tar'):
        print("Error: 'checkpoint_multi.pth.tar' not found.")
        exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-disp', action='store_true')
    args = parser.parse_args()

    random.seed(1)
    color_palette = train_seg_model.get_tableau_palette()

    # Note: Please don't change the order in object_shapes and object_meshes array.
    #   their order is consistent with the trained segmentation model.
    object_shapes = [
        "assets/objects/cube.urdf",
        "assets/objects/rod.urdf",
        "assets/objects/custom.urdf",
    ]
    object_meshes = [
        "assets/objects/cube.obj",
        "assets/objects/rod.obj",
        "assets/objects/custom.obj",
    ]
    env = sim.PyBulletSim(object_shapes = object_shapes, gui=args.disp)
    env.load_gripper()

    # setup camera (this should be consistent with the camera 
    #   used during training segmentation model)
    my_camera = camera.Camera(
        image_size=(480, 640),
        near=0.01,
        far=10.0,
        fov_w=50
    )
    camera_target_position = (env._workspace1_bounds[:, 0] + env._workspace1_bounds[:, 1]) / 2
    camera_target_position[2] = 0
    camera_distance = np.sqrt(((np.array([0.5, -0.5, 0.8]) - camera_target_position)**2).sum())
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target_position,
        distance=camera_distance,
        yaw=90,
        pitch=-60,
        roll=0,
        upAxisIndex=2,
    )

    # Prepare model (again, should be consistent with segmentation training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = 3  # RGB
    n_classes = len(object_shapes) + 1  # number of objects + 1 for background class
    model = train_seg_model.miniUNet(n_channels, n_classes)
    model.to(device)
    model, _, _ = train_seg_model.load_chkpt(model, 'checkpoint.pth.tar', device)
    model.eval()
    model.to(device)
    # Solution version:
    # ===============================================================================
    rgb_trans = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(train_seg_model.mean_rgb, train_seg_model.std_rgb),
    ])
    # ===============================================================================

    obj_ids = env._objects_body_ids  # everything else will be treated as background

    is_grasped = np.zeros(3).astype(np.bool)
    while not np.all(is_grasped):  # Keep repeating until the tote is empty
        # Capture rgb and depth image of the tote.
        rgb_obs, depth_obs, _ = camera.make_obs(my_camera, view_matrix)

        # TODO: now generate the segmentation prediction from the model
        # pred should contain the predicted segmentation mask
        # ==================================================================================
        dataset= ImgDataset(rgb_obs)
    
        loader = DataLoader(dataset, batch_size=1,
                        shuffle=False)
        inputs = next(iter(loader))
        inputs=inputs['input']
        inputs = Variable(inputs.to(device))
        with torch.no_grad():
            pred = model(inputs)
            _, pred = torch.max(pred, dim=1)
        # ==================================================================================

        markers = []
        # Points in each point cloud to use for ICP.
        #   Adjust this as per your machine performance.
        num_sample_pts = 100
        
        # Randomly choose an object index to grasp which is not grasped yet.
        # [optional] You can also try out some heuristics to choose which object to grasp.
        #    For example: grasp object which is most isolated to avoid collision with other objects
        obj_index = np.random.choice(np.where(~is_grasped)[0], 1)[0]
        #pred_image = train_seg_model.convert_seg_split_into_color_image(pred[0].cpu().numpy())
        # TODO: Mask out the depth based on predicted segmentation mask of object.
        # ====================================================================================
        #print(pred_image.shape)
        obj_depth =icp.gen_obj_depth(obj_index+1, depth_obs, pred[0])

        # ====================================================================================

        # TODO: transform depth to 3d points in camera frame. We will refer to these points as
        #   segmented point cloud or seg_pt_cloud.
        # ====================================================================================
        cam_pts = np.asarray(transforms.depth_to_point_cloud(my_camera.intrinsic_matrix, obj_depth))
        # ====================================================================================
        if cam_pts.shape == (0,):
            print("No points are present in segmented point cloud. Please check your code. Continuing ...")
            continue

        # TODO: transform 3d points (seg_pt_cloud) in camera frame to the world frame
        # ====================================================================================
        world_pts = transforms.transform_point3s(camera.cam_view2pose(view_matrix), cam_pts)
        # ====================================================================================

        world_pts_sample = world_pts[np.random.choice(range(world_pts.shape[0]), num_sample_pts), :]
        # (optional) Uncomment following to visualize points as small red spheres.
        #   These should approximately lie on chosen object index
        #for position in world_pts_sample:
            #markers.append(sim.SphereMarker(position=position, radius=0.001, rgba_color=[1, 0, 0, 0.8]))

        # Sample points from ground truth mesh. 
        # TODO: sample pts from known object mesh. Use object_shapes[obj_index]
        #   to locate path of the mesh.
        # - We will call these points ground truth point cloud or gt_pt_cloud.
        # - Hint: use icp.mesh2pts function from hw2
        # ====================================================================================
        gt_pt_cloud=icp.mesh2pts(object_meshes[obj_index], len(world_pts))
        # ====================================================================================
        # TODO: Align ground truth point cloud (gt_pt_cloud) to segmented 
        #   point cloud (seg_pt_cloud) using ICP.
        # - Hint: use icp.align_pts function from hw2
        #transform = None  # should contain the transformation matrix for transforming
        #  ground truth object point cloud to the segmented object point cloud.
        #transformed = None # should contain transformed ground truth point cloud
        # ====================================================================================
        transform, transformed= icp.align_pts(gt_pt_cloud, world_pts, 100, 1e-20)
        # ====================================================================================

        # (optional) Uncomment following to visualize transformed points as small black spheres.
        #   These should approximately lie on chosen object index
        #for position in transformed:
            #markers.append(sim.SphereMarker(position=position, radius=0.001, rgba_color=[0, 0, 0, 0.8]))
    
        # TODO: extract grasp position and angle
        #position = None  # This should contain the grasp position
        #grasp_angle = None  # This should contain the grasp angle
        # ====================================================================================
        #Use transform matrix
        r = Rotation.from_matrix(transform[:3, :3])
        grasp_angle= r.as_euler('zyx', degrees=False)#from scipy.spatial.transform import Rotation check documentation
        position=transform[:3, 3:4].flatten()

        # ====================================================================================

        # visualize grasp position using a big red sphere
        markers.append(sim.SphereMarker(position, radius = 0.02))



        # attempt grasping
        grasp_success = env.execute_grasp(position, grasp_angle[0])
        print(f"Grasp success: {grasp_success}")

        if grasp_success:  # Move the object to another tote
            is_grasped[obj_index] = True

            # Get a list of robot configurations in small step sizes
            path_conf = rrt(env.robot_home_joint_config,
                            env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.5, env)
            if path_conf is None:
                print("no collision-free path is found within the time budget. continuing ...")
            else:
                env.set_joint_positions(env.robot_home_joint_config)
                execute_path(path_conf, env)
        del markers
        p.removeAllUserDebugItems()
        env.robot_go_home()
        # env.reset_objects()