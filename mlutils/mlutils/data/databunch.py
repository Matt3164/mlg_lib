from dataclasses import dataclass
from typing import Optional

from mlutils.data.dataloader import Dataloader


@dataclass
class Databunch(object):
    train_dl: Dataloader
    test_dl: Dataloader
    valid_dl: Optional[Dataloader] = None
