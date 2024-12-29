import torch


def compute_ade(predicted, target):
    """
    Compute Average displacement error (ADE) and Minimum average displacement error (Min-ADE)
    """
    id_to_idx = {id_.item(): idx for idx, id_ in enumerate(target["id"])}
    ade = torch.tensor([])

    for agent_id in predicted:
        agent_idx = id_to_idx[agent_id]
        if target["target_mask"][agent_idx] or agent_idx == target["av_index"]:
            # shape: [32 simulated times, 80 steps, 3 dim]
            pred_trajectories = torch.from_numpy(predicted[agent_id])[..., :3]
            # [80 steps, 3 dim]
            target_trajectory = target["position"][agent_idx][11:]
            # target_trajectory = torch.cat((target["position"][agent_idx],
            #                                target["heading"][agent_idx].unsqueeze(-1)),
            #                               dim=-1)[11:]

            # [32 simulated times, valid steps]
            displacement_error = torch.norm(pred_trajectories - target_trajectory, p=2, dim=-1)
            valid_mask = target["valid_mask"][agent_idx][11:]
            displacement_error = displacement_error[:, valid_mask]

            # [32 simulated steps, 1]
            average_displacement_error = displacement_error.mean(-1).unsqueeze(-1)
            # [32 simulated steps, N]
            ade = torch.cat((ade, average_displacement_error), dim=-1)

    return ade.mean(), torch.min(ade.mean(dim=-1))
