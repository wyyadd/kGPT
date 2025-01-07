"""
Path tracking simulation with Stanley steering control and PID speed control.
"""
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from utils import wrap_angle


class KinematicControl:
    def __init__(self,
                 ref_x,
                 ref_y,
                 ref_yaw,
                 ref_v,
                 wheel_base,
                 dt=0.1,
                 speed_control=1.2,
                 max_steering_angle=torch.tensor(torch.pi / 2)):
        super().__init__()
        self.ref_x = ref_x - wheel_base * torch.cos(ref_yaw) / 2
        self.ref_y = ref_y - wheel_base * torch.sin(ref_yaw) / 2
        self.ref_yaw = ref_yaw
        self.ref_v = ref_v
        self.wheel_base = wheel_base
        self.x = self.ref_x[0].clone()
        self.y = self.ref_y[0].clone()
        self.yaw = self.ref_yaw[0].clone()
        self.v = self.ref_v[0].clone()

        self.delta = torch.zeros_like(ref_yaw)
        self.acc = torch.zeros_like(ref_yaw)

        self.dt = dt
        self.max_steering_angle = max_steering_angle
        self.speed_control = speed_control

    def update(self, acceleration, delta, wheel_base, idx):
        """
        Update the state of the vehicle. Stanley Control uses bicycle model.
        """
        delta = torch.clip(delta, -self.max_steering_angle, self.max_steering_angle)
        self.yaw += self.v / wheel_base * torch.tan(delta) * self.dt
        self.yaw = wrap_angle(self.yaw)

        self.v += acceleration * self.dt

        self.x += self.v * torch.cos(self.yaw) * self.dt
        self.y += self.v * torch.sin(self.yaw) * self.dt

        self.acc[idx - 1] = acceleration
        self.delta[idx - 1] = delta

    def get_control_actions(self):
        return self.acc, self.delta

    def calculate_pure_pursuit_control(self, target_idx):
        # speed control
        current_vector = torch.tensor([self.x - self.ref_x[target_idx], self.y - self.ref_y[target_idx]])
        current_d = torch.norm(current_vector, p=2, dim=-1)
        target_vector = torch.tensor([self.ref_x[target_idx] - self.ref_x[target_idx - 1],
                                      self.ref_y[target_idx] - self.ref_y[target_idx - 1]])
        target_d = torch.norm(target_vector, p=2, dim=-1)
        if current_d > target_d * self.speed_control:
            acc = self.speed_control * (self.ref_v[target_idx] - self.v) / self.dt
        else:
            acc = (self.ref_v[target_idx] - self.v) / self.dt

        # https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/PurePursuit.html
        alpha = torch.arctan2(self.ref_y[target_idx] - self.y, self.ref_x[target_idx] - self.x) - self.yaw
        # reset yaw if alpha is greater than 0.5pi
        if torch.abs(wrap_angle(alpha)) > torch.pi / 2:
            delta = torch.arctan2(self.wheel_base[target_idx - 1] *
                                  (wrap_angle(self.ref_yaw[target_idx] - self.yaw) / self.dt), self.v)
        else:
            delta = torch.arctan2(2.0 * self.wheel_base[target_idx - 1] * torch.sin(alpha), current_d)
        return acc, delta

    def step(self, target_idx):
        acc, delta = self.calculate_pure_pursuit_control(target_idx)
        self.update(acc, delta, self.wheel_base[target_idx - 1], target_idx)

    def run(self, show_animation=False):
        time = 0.0
        x = [self.x.item()]
        y = [self.y.item()]
        yaw = [self.yaw.item()]
        v = [self.v.item()]
        t = [0.0]

        for target_idx in range(1, len(self.ref_v)):
            self.step(target_idx)
            if show_animation:
                time += self.dt
                x.append(self.x.item())
                y.append(self.y.item())
                yaw.append(self.yaw.item())
                v.append(self.v.item())
                t.append(time)
                plt.cla()
                plt.plot(self.ref_x, self.ref_y, ".r", label="course")
                plt.plot(x, y, ".b", label="trajectory")
                plt.plot(self.ref_x[target_idx], self.ref_y[target_idx], "xg", label="target")
                plt.axis("equal")
                plt.grid(True)
                plt.title(f"Speed[m/s]:{self.v.item():.2f}, steps:{target_idx}, yaw:{self.yaw.item():.2f}")
                plt.pause(0.001)

        if show_animation:
            plt.plot(self.ref_x, self.ref_y, ".r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.legend()
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.axis("equal")
            plt.grid(True)

            plt.subplots(1)
            plt.plot(t, v, "-r")
            plt.xlabel("Time[s]")
            plt.ylabel("Speed[m/s]")
            plt.grid(True)
            plt.show()


def get_control_actions(scenario: HeteroData) -> HeteroData:
    delta_time = 0.1

    acceleration = torch.zeros_like(scenario["agent"]["heading"])
    delta = torch.zeros_like(scenario["agent"]["heading"])

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
            if end - start == 1:
                continue
            length = scenario["agent"]["length"][agent_idx, start:end]
            cyaw = scenario["agent"]["heading"][agent_idx, start:end]
            cx = scenario["agent"]["position"][agent_idx, start:end, 0]
            cy = scenario["agent"]["position"][agent_idx, start:end, 1]

            cv = torch.zeros_like(cyaw)
            temp = scenario["agent"]["position"][agent_idx, start:end, :2]
            cv[1:] = torch.norm(temp[1:] - temp[:-1], p=2, dim=-1) / delta_time
            cv[0] = torch.hypot(scenario["agent"]["velocity"][agent_idx, start, 0],
                                scenario["agent"]["velocity"][agent_idx, start, 1])

            control = KinematicControl(cx, cy, cyaw, cv, length, dt=delta_time)
            control.run(show_animation=False)
            a, d = control.get_control_actions()
            acceleration[agent_idx, start:end] = a
            delta[agent_idx, start:end] = d
    scenario["agent"]["acceleration"] = acceleration
    scenario["agent"]["delta"] = delta
    return scenario


class ControlActionBuilder(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, scenario: HeteroData) -> HeteroData:
        return get_control_actions(scenario)
