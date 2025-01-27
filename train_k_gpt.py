from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.loader import DataLoader

from datamodules import WaymoSimDataModule
from datasets import WaymoSimDataset
from simulators import KGPT
from transforms import SimAgentFilter

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(2025, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--submission_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--simulation_times', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=10)
    parser.add_argument('--grad_batch_size', type=int, default=1)
    KGPT.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)

    if args.mode == 'train':
        model = KGPT(**vars(args))
        datamodule = WaymoSimDataModule(**vars(args))
        model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=5, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, num_nodes=args.num_nodes,
                             strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
                             callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs,
                             precision="32-true", accumulate_grad_batches=args.grad_batch_size)
        trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)
    else:
        model = KGPT.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            simulation_times=args.simulation_times,
            submission_dir=args.submission_dir)
        test_dataset = WaymoSimDataset(root=args.root, split=args.mode, submission_dir=args.submission_dir,
                                       interactive=args.interactive, transform=SimAgentFilter(1024))
        dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                shuffle=False, pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
        trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                             strategy="ddp", num_nodes=args.num_nodes)
        trainer.test(model, dataloader)
