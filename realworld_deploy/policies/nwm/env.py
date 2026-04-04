import importlib
import numpy as np
from PIL import Image
import torch
from discoverse.examples.mmk2.navigation.move_to_point_mujoco import MoveToPointV3
import mujoco
import math
import os
from utils import save_img

class Env():
    def __init__(self, args, imgTransform):
        module = importlib.import_module(args.task_path.replace("/", ".").replace(".py", ""))
        SimNode = getattr(module, "SimNode") # SimNode
        cfg = getattr(module, "cfg") # cfg
        self.cfg = cfg
        self.simnode = SimNode(cfg)
        self.args = args
        self.obs_steps = args.obs_steps
        self.obs_que = None
        self.video_list = list()
        self.step_record = list()   # 每次调用step函数的录像，每次调用前清空
        self.imgTransform = imgTransform
        self.blink = True

    def reset(self):
        goal = self._set_goal()
        obs, t = self.simnode.reset(), 0
        if self.args.random_origin:
            obs, _, _ = self.simnode.random_set_base()
        else:
            obs = self.simnode.spec_set_base(self.args.origin_position, self.args.origin_orientation)
        self.video_list = list()
        from collections import  deque
        self.obs_que = deque([obs], maxlen=self.obs_steps+1)
        init_goal_dist = self.simnode.get_goal_distance()
        return self.obs_que_ext(), t, goal, init_goal_dist
    
    def obs_que_ext(self):
        result = dict()
        # 处理所有非图像类型的观测
        for key in self.args.obs_keys:
            if not key.startswith('image'):
                result[key] = self.stack_last_n_obs(
                    [np.array(obs[key]) for obs in self.obs_que]
                )
        # 处理图像类型的观测
        image_ids = [int(key[5:]) for key in self.args.obs_keys if key.startswith('image')]
        imgs = {id: [] for id in image_ids}
        for obs in self.obs_que:
            img = obs['img']
            for id in image_ids:
                #img_trans = np.transpose(img[id] / 255, (2, 0, 1))  # 转置并归一化
                img_trans = self.imgTransform(Image.fromarray(img[id], mode='RGB'))
                imgs[id].append(img_trans)

        for id in image_ids:
            result[f'image{id}'] = self.stack_last_n_obs(imgs[id])
        return result
    
    def step(self, action):
        success = 0
        self.step_record = list()
        for i, act in enumerate(action): #依次执行每个动作
            #print("move step%i..." %i)
            cur_state = self.simnode.getObservation()
            cur_position = cur_state['base_position']
            cur_orientation = cur_state['base_orientation']

            R_initial = np.zeros(9)
            mujoco.mju_quat2Mat(R_initial, cur_orientation)
            R_initial = R_initial.reshape(3, 3)
            forward_direction = R_initial[:, 0]
            cur_theta = np.rad2deg(math.atan2(forward_direction[1], forward_direction[0]))

            # 变换为世界坐标
            traget_pos = R_initial.dot(np.concatenate([act[:2], [0]])) + cur_position

            move_to_point = MoveToPointV3({"x": cur_position[0], "y": cur_position[1], "theta": cur_theta}, {
                "x": traget_pos[0], "y": traget_pos[1], "theta": cur_theta + act[2]}, self.simnode)

            if self.blink:
                move_success = move_to_point.blink_to_point()
            else:
                move_success = move_to_point.move_to_point_no_turn_back()

            obs = self.simnode.getObservation()
            self.obs_que.append(obs) #添加单个obs
            self.video_list.append(obs['img'])
            self.step_record.append(obs['img'])

        if self.simnode.check_success():
            success = 1
        metric_data = {"ne": self.simnode.get_goal_distance()}

        return self.obs_que_ext(), success, move_success, metric_data
       
    def stack_last_n_obs(self, all_obs):
        assert(len(all_obs) > 0)
        result = torch.zeros((self.obs_steps,) + all_obs[-1].shape,
            dtype=all_obs[-1].dtype)
        start_idx = -min(self.obs_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:])
        if self.obs_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
        return result

    def _set_goal(self):
        if self.args.random_goalimg:
            obs, goal_position, goal_orientation = self.simnode.random_set_base()
            self.simnode.set_goal(goal_position, goal_orientation)
            img = obs["img"]
        else:
            goal_position = self.args.spec_position
            goal_orientation = self.args.spec_orientation
            img = self.simnode.spec_set_base(goal_position, goal_orientation)["img"]
            self.simnode.set_goal(goal_position, goal_orientation)
        return img

