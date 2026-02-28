import torch.utils.data
from torch.utils.data.dataloader import default_collate

from batch import BatchSubstructContext, BatchMasking, BatchAE

class DataLoaderSubstructContext(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext.from_data_list(data_list),
            **kwargs)

class DataLoaderMasking(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasking.from_data_list(data_list),
            **kwargs)


from util import MaskAtom

class DataLoaderMaskingPred(torch.utils.data.DataLoader):

    r"""
    数据加载器，用于将来自 torch_geometric.data.Dataset 的数据对象合并为小批量（mini-batch）。
        参数：
        dataset (Dataset): 要从中加载数据的数据集。
        batch_size (int, 可选): 每个批次加载的样本数量。（默认值：1）
        shuffle (bool, 可选): 如果设置为 True，数据将在每个 epoch（训练轮次）开始时重新打乱。（默认值：True）
    """
    def __init__(self, dataset, batch_size=1, shuffle=True,
                 mask_rate=0.0, mask_edge=0.0, **kwargs):

        self._transform = MaskAtom(
            num_atom_type=119,
            num_edge_type=5,
            mask_rate=mask_rate,
            mask_edge=mask_edge
        )
        # 使用自定义 collate_fn 替换默认 batch 拼接函数
        super(DataLoaderMaskingPred, self).__init__(
            dataset,
            batch_size,
            shuffle,
            # 指定自定义拼接函数
            collate_fn=self.collate_fn,
            # 其余参数透传给 DataLoader
            **kwargs
        )
    # 自定义 batch 拼接函数
    # DataLoader 会自动从 dataset 里取出 batch_size 个元素（通常是图对象 Data）。
    # 这 batch_size 个元素会组成一个 list，然后被传给 collate_fn 的参数 batches。
    def collate_fn(self, batches):
        # 对 batch 中每个图做mask
        batchs = [self._transform(x) for x in batches]
        # 把多个图拼成一个 batch 图对象
        return BatchMasking.from_data_list(batchs)

# class DataLoaderMaskingPred(torch.utils.data.DataLoader):
#     r"""Data loader which merges data objects from a
#     :class:`torch_geometric.data.dataset` to a mini-batch.
#     Args:
#         dataset (Dataset): The dataset from which to load the data.
#         batch_size (int, optional): How may samples per batch to load.
#             (default: :obj:`1`)
#         shuffle (bool, optional): If set to :obj:`True`, the data will be
#             reshuffled at every epoch (default: :obj:`True`)
#     """

#     def __init__(self, dataset, batch_size=1, shuffle=True, mask_rate=0.0, mask_edge=0.0, **kwargs):
#         self._transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = mask_rate, mask_edge=mask_edge)
#         super(DataLoaderMaskingPred, self).__init__(
#             dataset,
#             batch_size,
#             shuffle,
#             collate_fn=self.collate_fn,
#             **kwargs)
    
#     def collate_fn(self, batches):
#         batchs = [self._transform(x) for x in batches]
#         return BatchMasking.from_data_list(batchs)


class DataLoaderAE(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderAE, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchAE.from_data_list(data_list),
            **kwargs)



