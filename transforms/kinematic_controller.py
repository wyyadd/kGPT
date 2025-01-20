"""
Path tracking simulation with Stanley steering control and PID speed control.
"""
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Union, Dict

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from utils import wrap_angle


class KinematicControl:
    def __init__(self,
                 ref_x,
                 ref_y,
                 ref_yaw,
                 ref_v,
                 wheel_base,
                 dt=0.1):
        super().__init__()
        self.ref_x = ref_x
        self.ref_y = ref_y
        self.ref_yaw = ref_yaw
        self.ref_v = ref_v
        self.wheel_base = wheel_base
        self.x = self.ref_x[0].clone()
        self.y = self.ref_y[0].clone()
        self.yaw = self.ref_yaw[0].clone()
        self.v = ref_v.clone()

        self.delta = torch.zeros_like(ref_yaw)
        self.acc = torch.zeros_like(ref_yaw)

        self.dt = dt

        self.de = torch.tensor([])

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle. Stanley Control uses bicycle model.
        """
        # delta = torch.clip(delta, -self.max_steering_angle, self.max_steering_angle)
        # self.yaw += self.v / wheel_base * torch.tan(delta) * self.dt
        self.yaw += delta * self.dt
        self.yaw = wrap_angle(self.yaw)

        self.v += acceleration * self.dt

        self.x += self.v * torch.cos(self.yaw) * self.dt
        self.y += self.v * torch.sin(self.yaw) * self.dt

    def get_control_actions(self):
        return self.acc, self.delta

    def calculate_pure_pursuit_control(self, target_idx):
        current_d = torch.hypot(self.x - self.ref_x[target_idx], self.y - self.ref_y[target_idx])
        # https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/PurePursuit.html
        alpha = torch.arctan2(self.ref_y[target_idx] - self.y, self.ref_x[target_idx] - self.x) - self.yaw
        delta = torch.arctan2(2.0 * self.wheel_base[target_idx - 1] * torch.sin(alpha), current_d)
        return delta

    def calculate_direct_control(self, target_idx):
        delta_d = torch.hypot(self.ref_x[target_idx] - self.x, self.ref_y[target_idx] - self.y)
        acc = (delta_d / self.dt - self.v) / self.dt
        yaw = torch.arctan2(self.ref_y[target_idx] - self.y, self.ref_x[target_idx] - self.x)
        delta = (yaw - self.yaw) / self.dt
        if delta_d < 0.03 and torch.abs(delta) > torch.pi / 2:
            acc = -self.v / self.dt
            delta = 0
        # else:
        #     print(target_idx, delta_d, yaw - self.yaw)
        return acc, delta

    def step(self, target_idx):
        acc, delta = self.calculate_direct_control(target_idx)
        self.acc[target_idx] = acc
        self.delta[target_idx] = delta
        self.update(acc, delta)

    def run(self, show_animation=False, generate_metrics=False):
        time = 0.0
        x = [self.x.item()]
        y = [self.y.item()]
        yaw = [self.yaw.item()]
        v = [self.v.item()]
        t = [0.0]
        for target_idx in range(1, len(self.ref_yaw)):
            self.step(target_idx)

            if show_animation or generate_metrics:
                time += self.dt
                x.append(self.x.item())
                y.append(self.y.item())
                yaw.append(self.yaw.item())
                v.append(self.v.item())
                t.append(time)

            if show_animation:
                plt.cla()
                plt.plot(self.ref_x, self.ref_y, ".r", label="course")
                plt.plot(x, y, ".b", label="trajectory")
                plt.plot(self.ref_x[target_idx], self.ref_y[target_idx], "xg", label="target")
                plt.axis("equal")
                plt.grid(True)
                plt.title(f"Speed[m/s]:{self.v.item():.2f}, steps:{target_idx}, yaw:{self.yaw.item():.2f}")
                plt.pause(0.001)

        if generate_metrics:
            de = torch.norm(torch.stack([self.ref_x, self.ref_y, self.ref_yaw], dim=-1) - torch.stack(
                [torch.tensor(x), torch.tensor(y), torch.tensor(yaw)], dim=-1), p=2, dim=-1)
            self.de = de.mean(-1).unsqueeze(-1)

        if show_animation:
            plt.plot(self.ref_x, self.ref_y, ".r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.legend()
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.axis("equal")
            plt.grid(True)

            plt.subplots(1)
            plt.title("red: velocity, blue: acc, green: real_v")
            plt.plot(t, v, "-r")
            plt.plot(t, self.acc, "-b")
            plt.plot(t, self.ref_v, "-g")
            plt.xlabel("Time[s]")
            plt.grid(True)

            plt.subplots(1)
            plt.title("red: yaw, blue: angular_v, green: real_yaw")
            plt.plot(t, yaw, "-r")
            plt.plot(t, self.delta, "-b")
            plt.plot(t, self.ref_yaw, "-g")
            plt.xlabel("Time[s]")
            plt.grid(True)
            plt.show()


def get_control_actions(scenario: Union[HeteroData, str, Dict], show_animation=False, generate_metrics=False):
    if isinstance(scenario, str):
        with open(scenario, "rb") as f:
            scenario = pickle.load(f)

    delta_time = 0.1
    acceleration = torch.zeros_like(scenario["agent"]["heading"])
    delta = torch.zeros_like(scenario["agent"]["heading"])
    ade = torch.tensor([])

    for agent_idx in range(scenario["agent"]["num_nodes"]):
        valid_mask = scenario["agent"]["valid_mask"][agent_idx]
        mask_diff = torch.diff(valid_mask.int())
        segment_starts = (mask_diff == 1).nonzero(as_tuple=True)[0] + 1
        segment_ends = (mask_diff == -1).nonzero(as_tuple=True)[0] + 1
        if valid_mask[0]:
            segment_starts = torch.cat((torch.tensor([0]), segment_starts))
        if valid_mask[-1]:
            segment_ends = torch.cat((segment_ends, torch.tensor([valid_mask.size(0)])))

        for start, end in zip(segment_starts, segment_ends):
            if end - start <= 1:
                continue
            length = scenario["agent"]["length"][agent_idx, start:end]
            cyaw = scenario["agent"]["heading"][agent_idx, start:end]
            cx = scenario["agent"]["position"][agent_idx, start:end, 0]
            cy = scenario["agent"]["position"][agent_idx, start:end, 1]

            if (cx == cx[0]).all() and (cy == cy[0]).all():
                continue

            cv = torch.hypot(scenario["agent"]["velocity"][agent_idx, start, 0],
                             scenario["agent"]["velocity"][agent_idx, start, 1])

            # cv = torch.zeros_like(cyaw)
            # temp = scenario["agent"]["position"][agent_idx, start:end, :2]
            # cv[1:] = torch.norm(temp[1:] - temp[:-1], p=2, dim=-1) / delta_time
            # cv[0] = torch.hypot(scenario["agent"]["velocity"][agent_idx, start, 0],
            #                     scenario["agent"]["velocity"][agent_idx, start, 1])

            control = KinematicControl(cx, cy, cyaw, cv, length, dt=delta_time)
            control.run(show_animation, generate_metrics)
            a, d = control.get_control_actions()
            acceleration[agent_idx, start:end] = a
            delta[agent_idx, start:end] = d
            ade = torch.cat((ade, control.de), dim=-1)
    return acceleration, delta, ade.mean()


class ControlActionBuilder(BaseTransform):
    def __init__(self, patch: int = 10) -> None:
        super().__init__()
        self.patch = patch

    def __call__(self, data: HeteroData) -> HeteroData:
        pos = data['agent']['position']
        target_idx = data['agent']['target_idx']
        # num_agent, steps, patch_size, 3 -> acceleration, delta, height
        data['agent']['target'] = pos.new_zeros(target_idx.numel(), pos.shape[1], self.patch, 3)
        delta_height = pos[:, 1:, 2] - pos[:, :-1, 2]
        acc = data['agent']['acc']
        delta = data['agent']['delta']
        # acc, delta, _ = get_control_actions(data)
        for t in range(self.patch):
            data['agent']['target'][:, :-t - 1, t, 0] = acc[target_idx, t + 1:]
            data['agent']['target'][:, :-t - 1, t, 1] = delta[target_idx, t + 1:]
            data['agent']['target'][:, :-t - 1, t, 2] = delta_height[target_idx, t:]
        return data


def calculate_metrics():
    target_folder = "../data/validation/processed"
    predicted_files = [name for name in os.listdir(target_folder)]
    total_ade = torch.tensor([])
    with ProcessPoolExecutor() as executor:
        futures = []
        for pkl_file in predicted_files:
            futures.append(executor.submit(get_control_actions, os.path.join(target_folder, pkl_file), False, True))

        for future in tqdm(as_completed(futures), total=len(futures)):
            _, _, ade = future.result()
            total_ade = torch.cat((total_ade, ade.unsqueeze(-1)), dim=0)

    with open("ade.pkl", "wb") as f:
        pickle.dump(total_ade, f)
    print(total_ade.shape, total_ade.mean())
