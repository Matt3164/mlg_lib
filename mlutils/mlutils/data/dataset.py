class Dataset(object):
    def __getitem__(self, item: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError