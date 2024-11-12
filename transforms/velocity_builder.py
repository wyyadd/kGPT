from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class VelocityBuilder(BaseTransform):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: HeteroData) -> HeteroData:
        t = 0.1
        pos = data['agent']['position']
        vel = data['agent']['velocity']
        # generate new velocity
        # [num_nodes, steps, vel_dimension-x,y,z]
        new_vel = vel.new_zeros(data['agent']['num_nodes'], pos.size(-2), 3)
        new_vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / t
        new_vel[:, 0, :2] = vel[:, 0, :2]
        new_vel[:, 0, 2] = new_vel[:, 1, 2]
        data['agent']['velocity'] = new_vel

        return data
