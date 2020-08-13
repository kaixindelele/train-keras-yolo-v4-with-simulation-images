from modder import TextureModder
import argparse
import numpy as np
import os
import time
import tensorflow as tf
import pandas as pd
import random
import robosuite as suite

from connect_dataset.array2voc_class import Array2voc

from algorithm.reinforce_algs.td3_sp.TD3_class_well import TD3
from algorithm.experiments_utils.experiments_utils import try_make_dir
from algorithm.experiments_utils.noise.ou_noise import OU_noise
from algorithm.experiments_utils.sava_data_set import SaveDataSet
from algorithm.experiments_utils.object_or_table_utils import is_cube_on_table
from algorithm.experiments_utils.sp_utils.logx import EpochLogger
from robosuite.utils.transforms3d.euler import euler2quat,quat2euler
from math import pi
from algorithm.experiments_utils.shelter_utils_lyl_sixpoints_totalarean import *
from skimage.transform import resize
from robosuite.utils.camera_transform import CameraTransform
from robosuite.utils.transform_utils import convert_quat
# 要识别颜色的下限　默认红色
Lower = np.array([0, 43, 46])
Upper = np.array([10, 255, 255])


def rgb2hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    m = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        if g >= b:
            h = ((g - b) / m) * 60
        else:
            h = ((g - b) / m) * 60 + 360
    elif mx == g:
        h = ((b - r) / m) * 60 + 120
    elif mx == b:
        h = ((r - g) / m) * 60 + 240
    if mx == 0:
        s = 0
    else:
        s = m / mx
    v = mx
    H = h / 2
    S = s * 255.0
    V = v * 255.0
    return H, S, V


def get_nored():
    while True:
        rgb = [np.random.randint(255), np.random.randint(255), np.random.randint(255), 1]
        hsv = rgb2hsv(rgb[0], rgb[1], rgb[2])
        if hsv[0] <= Upper[0] and hsv[0] >= Lower[0] \
            and hsv[1] <= Upper[1] and hsv[1] >= Lower[1] \
            and hsv[2] <= Upper[2] and hsv[2] >= Lower[2]:
            pass
        else:
            break
    return rgb


def move_objects(env, obj_name, state):
    """
    move objects with name @obj out of the task space. This is useful
    for supporting task modes with single types of objects, as in
    @env.single_object_mode without changing the model definition.
    """

    addr = env.sim.model.get_joint_qpos_addr(obj_name)[0]

    sim_state = env.sim.get_state()
    # sim_state.qpos[env.sim.model.get_joint_qpos_addr(obj_name)[0]] = state
    sim_state.qpos[addr+0] = state[0]
    sim_state.qpos[addr+1] = state[1]
    sim_state.qpos[addr+2] = state[2]
    # sim_state.qpos[addr+3] = random.uniform(-pi*90/180,pi*90/180)

    env.sim.set_state(sim_state)
    env.sim.forward()


def run(args, logs_path=None):
    np.random.seed(args.seed)
    random.seed(args.seed)
    array2voc = Array2voc()

    """模块一：创建仿真环境"""
    env = suite.make(
        args.env_name,
        has_renderer=args.has_renderer,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=10,
        gripper_type=args.gripper_type,
        camera_depth=False,
        render_visual_mesh=True,
        reward_shaping=True,
        camera_height=args.image_height,
        camera_width=args.image_width,
        camera_name="singlecam",
    )
    obs = env.reset()

    light_ids = []
    for i in range(1, 7):
        id = env.sim.model.light_name2id("light%d" % i)
        light_ids.append(id)
        env.sim.model.light_castshadow[id] = 1  # has castshadow
        env.sim.model.light_active[id] = 1

    """模块二：创建强化学习算法"""
    s_dim = obs['joint_pos'].shape[0] + obs['eef_pos'].shape[0] + \
            obs['robot-state'].shape[0] + obs['object-state'].shape[0]
    a_dim = np.array(env.action_spec).shape[1]
    action_low_bound, action_high_bound = env.action_spec[0], env.action_spec[1]
    a_bound = action_high_bound
    net = TD3(a_dim, s_dim, a_bound,
              batch_size=args.batch_size,
              sess_opt=0.2,
              )

    """开始训练"""
    cube_size = [0.025, 0.025, 0.025]

    cube_id = env.sim.model.body_name2id("cube")
    camid = env.sim.model.camera_name2id("singlecam")

    table_geom_id = env.sim.model.geom_name2id("table_visual")
    base_table_geom_id = env.sim.model.geom_name2id("base_table_visual")
    white_table_geom_id = env.sim.model.geom_name2id("white_table")
    floor_geom_id = env.sim.model.geom_name2id("floor")

    env.sim.model.geom_rgba[table_geom_id] = [0, 1, 0, 1]
    env.sim.model.geom_rgba[white_table_geom_id] = [1, 1, 1, 1]
    env.sim.model.geom_rgba[white_table_geom_id] = [1, 1, 1, 1]

    scene_num = 0

    for i in range(args.max_episode):
        # 仿真环境改变
        obs = env.reset()
        modder = TextureModder(env.sim)
        cube_pos = env.sim.model.body_pos[cube_id]
        print("cube_pos_reset:", cube_pos)

        has_light = False
        for tt in range(6):
            light_id = light_ids[tt]
            env.sim.model.light_active[light_id] = np.random.randint(0, 1)
            if env.sim.model.light_active[light_id]:
                has_light = True
                env.sim.model.light_pos[light_id, 0] += random.uniform(-0.2, 0.2)
                env.sim.model.light_pos[light_id, 1] += random.uniform(-0.2, 0.2)
                brightness = random.uniform(0.4, 1.0)
                env.sim.model.light_diffuse[light_id] = [brightness, brightness, brightness]

        if has_light == False:
            env.sim.model.light_active[light_ids[np.random.randint(0, 6)]] = 1

        table_rgb = get_nored()
        env.sim.model.geom_rgba[table_geom_id] = [table_rgb[0] / 255, table_rgb[1] / 255, table_rgb[2] / 255, 1]

        base_table_rgb = get_nored()
        env.sim.model.geom_rgba[base_table_geom_id] = [base_table_rgb[0] / 255, base_table_rgb[1] / 255,
                                                       base_table_rgb[2] / 255, 1]
        floor_rgb = get_nored()
        env.sim.model.geom_rgba[floor_geom_id] = [floor_rgb[0] / 255, floor_rgb[1] / 255, floor_rgb[2] / 255, 1]
        modder.rand_checkboard(names=['table_visual', 'floor', 'white_table'])
        distance = random.uniform(args.min_distance, args.max_distance)
        theta = random.uniform(args.min_theta, args.max_theta)
        alpha = random.uniform(args.min_alpha, args.max_alpha)
        rotate = random.uniform(args.min_rotate, args.max_rotate)

        env.reset_singlecam(theta=theta, alpha=alpha, telescopic=distance, rotate=rotate)  # 0-60,60-120,1.3-2
        env.sim.model.cam_fovy[camid] = random.uniform(args.min_fovy,
                                                       args.max_fovy)

        s = np.hstack((obs['eef_pos'],
                       obs['joint_pos'],
                       obs['robot-state'],
                       obs['object-state']))

        for j in range(args.max_step):
            scene_num += 1
            # for t in range(1, 5):
            if j % 1 == 0:
                state = [-0.625+random.uniform(-0.25, 0.25),
                         random.uniform(-0.3, 0.3), 0.035]
                move_objects(env, 'cube', state)
            if i % 2 == 0:
                a = np.random.randn(env.dof)
            else:
                a = net.get_action(s, args.noise_scale)
            try:
                obs, r, done, info = env.step(a)
                # env.render()
            except Exception as e:
                print("step_e", e)
            s_ = np.hstack((obs['eef_pos'],
                            obs['joint_pos'],
                            obs['robot-state'],
                            obs['object-state']))
            net.store_transition((s, a, r, s_, done))
            s = s_

            object_default_pos = env.sim.data.body_xpos[cube_id]
            cube_pos = np.array(object_default_pos,
                                copy=True)

            ontable_flag = is_cube_on_table(cube_pos, cube_size,
                                            sim=env.sim)

            if ontable_flag == False:
                continue
            img = env.sim.render(
                camera_name="singlecam",
                width=env.camera_width,
                height=env.camera_height,
                depth=env.camera_depth,
            )
            img_true = np.flip(img.transpose((1, 0, 2)), 1)
            image = img_true[:, :, (2, 1, 0)]
            image_save = np.flip(image, 1)
            came_pos = env.sim.data.get_body_xpos("singlecam")
            came_quat = env.sim.data.get_body_xquat("singlecam")
            cube_quat_xyzw = convert_quat(np.array(env.sim.data.body_xquat[cube_id]),
                                          to="xyzw")

            camera_transform = CameraTransform(width=env.camera_width,
                                               height=env.camera_height,
                                               came_pos=came_pos,
                                               came_quat=came_quat,
                                               fov_angle=env.sim.model.cam_fovy[camid])
            center_pos_in_pixel = camera_transform.get_rgb(cube_pos)
            show_img = image_save.copy()
            cv2.circle(show_img,
                       (args.image_height - int(center_pos_in_pixel[1]),
                        int(center_pos_in_pixel[0])),
                       3, (255, 255, 100), 0)
            cv2.imshow("center_pos", show_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # # save mistake,so you need to transform to true format
            eight_points_pos_in_world = get_cube_eight_points(cube_pos,
                                                              cube_quat_xyzw,
                                                              cube_size)
            eight_points_pos_in_pixel = [camera_transform.get_rgb(pos) for pos in eight_points_pos_in_world]
            six_points_pos_in_pixel = get_six_points(eight_points_pos_in_pixel,
                                                     center_pos_in_pixel)
            # print('six_points_pos_in_pixel:',
            #       six_points_pos_in_pixel)
            show_img = image_save.copy()
            for point in six_points_pos_in_pixel:
                cv2.circle(show_img,
                           # (int(point[1]), int(point[0])),
                           (args.image_height - int(point[1]),
                            int(point[0])),
                           2, (255, 255, 0), 0)
            cv2.imshow("show_img", show_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            color_rate = get_color_rate_lyl_sixpoints_totalarea(img_true,
                                                                cube_pos,
                                                                cube_quat_xyzw,
                                                                cube_size,
                                                                camera_transform)

            if color_rate < 0.3:
                continue

            img_name = "image%d_%d.jpg" % (i, j)
            xmin = six_points_pos_in_pixel[:, 1].min()
            xmax = six_points_pos_in_pixel[:, 1].max()
            ymin = six_points_pos_in_pixel[:, 0].min()
            ymax = six_points_pos_in_pixel[:, 0].max()

            xmin = args.image_height - xmin
            xmax = args.image_height - xmax
            # ymin = args.image_height - ymin
            # ymax = args.image_height - ymax

            draw_1 = cv2.rectangle(image_save,
                                   (xmin, ymin),
                                   (xmax, ymax),
                                   (222, 255, 222), 2)
            # cv2.rectangle(draw_1,
            #               (xmin, ymin),
            #               (xmax, ymax),
            #               (222, 255, 222), 5)
            print("(xmin, ymin), (xmax, ymax):",
                  (xmin, ymin), (xmax, ymax),
                  )
            cv2.imshow('img', draw_1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            labels = np.array([[xmin, ymin, xmax, ymax, 'cube'],
                               ])
            array2voc.save_one_img(img=image_save,
                                   labels=labels,
                                   img_name=img_name)


def main():
    # 设置传参和默认值
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default="Scr5LiftReal")
    parser.add_argument('--gripper_type', type=str, default="TwoFingerGripperScr5")
    parser.add_argument('--image_height', type=int, default=512)
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--has_renderer', type=bool, default=True)
    parser.add_argument('--tablesize', type=str, default="80_80_80")
    parser.add_argument('--cubesize', type=str, default="5")
    parser.add_argument('--net_name', type=str, default='td3')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_episode', type=str, default=5)
    parser.add_argument('--max_step', type=int, default=4)
    parser.add_argument('--train_max', type=int, default=2)
    parser.add_argument('-per_step_change',type=float,default=2)
    parser.add_argument('--verification_ratio', type=float, default=0.1)
    parser.add_argument('--random_action_rate', type=float, default=0.4)
    parser.add_argument('--no_noise_start_eposide', type=str, default=3)
    parser.add_argument('--out_noise_dt', type=int, default=0.001)
    parser.add_argument('--out_noise_max_sigma', type=int, default=0.18)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--state_simple_or_full', type=str, default="full")
    parser.add_argument('--noise_type', type=str, default="OU")

    noise_value = 0.26
    parser.add_argument('--noise_scale', type=float, default=noise_value)

    random_seed = int(time.time() * 10000 % 10000)
    # random_seed = 0
    parser.add_argument('--seed', '-s', type=int, default=random_seed)
    parser.add_argument('-init_Actor_lr', type=float, default=0.001)
    parser.add_argument('-init_Critic_lr', type=float, default=0.002)

    parser.add_argument('-min_distance', type=float, default=0.45)
    parser.add_argument('-max_distance', type=float, default=2.35)

    parser.add_argument('-min_theta', type=float, default=-50)
    parser.add_argument('-max_theta', type=float, default=50)

    parser.add_argument('-min_alpha',type=float,default=-60)
    parser.add_argument('-max_alpha',type=float,default=60)

    parser.add_argument('-min_rotate',type=float,default=-40)
    parser.add_argument('-max_rotate',type=float,default=80)

    parser.add_argument('-min_fovy',type=float,default=30)
    parser.add_argument('-max_fovy',type=float,default=80)

    args = parser.parse_args()
    run(args, None)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()


