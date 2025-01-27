# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from datasets import WaymoSimDataset
from transforms import SimAgentFilter, ControlActionBuilder


class WaymoSimDataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 interactive: bool,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_raw_dir: Optional[str] = None,
                 val_raw_dir: Optional[str] = None,
                 test_raw_dir: Optional[str] = None,
                 train_processed_dir: Optional[str] = None,
                 val_processed_dir: Optional[str] = None,
                 test_processed_dir: Optional[str] = None,
                 patch_size: int = 5,
                 **kwargs) -> None:
        super(WaymoSimDataModule, self).__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.root = root
        self.interactive = interactive
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.patch_size = patch_size
        self.train_transform = Compose([SimAgentFilter(64), ControlActionBuilder(patch_size)])
        self.val_transform = Compose([SimAgentFilter(1024), ControlActionBuilder(patch_size)])
        self.test_transform = SimAgentFilter(1024)

    def prepare_data(self) -> None:
        WaymoSimDataset(self.root, 'train', self.interactive, self.train_raw_dir, self.train_processed_dir,
                        self.train_transform)
        WaymoSimDataset(self.root, 'val', self.interactive, self.val_raw_dir, self.val_processed_dir,
                        self.val_transform)
        WaymoSimDataset(self.root, 'test', self.interactive, self.test_raw_dir, self.test_processed_dir,
                        self.test_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = WaymoSimDataset(self.root, 'train', self.interactive, self.train_raw_dir,
                                             self.train_processed_dir, self.train_transform)
        self.val_dataset = WaymoSimDataset(self.root, 'val', self.interactive, self.val_raw_dir, self.val_processed_dir,
                                           self.val_transform)
        self.test_dataset = WaymoSimDataset(self.root, 'test', self.interactive, self.test_raw_dir,
                                            self.test_processed_dir, self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
