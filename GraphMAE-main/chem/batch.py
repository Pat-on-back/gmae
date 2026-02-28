import torch
from torch_geometric.data import Data, Batch

# 定义一个新的批图类 BatchMasking，继承自 Data
class BatchMasking(Data):

    r"""
        这是一个普通的 Python 对象，用于将一批图（batch of graphs）建模为一个大型的（不连通的）图。
        由于它以 :class:torch_geometric.data.Data 为基类，因此 Data 类的所有方法在这里也都可用。
        此外，可以通过分配向量 :obj:batch 来重建单个图，该向量将每个节点映射到其所属图的标识符。
    """
    # 构造函数
    def __init__(self, batch=None, **kwargs):

        # 调用父类 Data 的初始化
        # kwargs 中通常包含 edge_index, x 等图属性
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    # 静态方法：从 Data 列表构造 BatchMasking
    @staticmethod
    def from_data_list(data_list):

        r"""
        从包含 :class:torch_geometric.data.Data 对象的 Python 列表中构建一个批次（Batch）对象。
        分配向量 :obj:batch 将在构建过程中动态生成。
        """
        # 每个 data.keys 是该图拥有的属性名集合
        keys = [set(data.keys) for data in data_list]
        # 合并所有图的字段名
        keys = list(set.union(*keys))
        # 确保没有已有 batch 字段（避免冲突）
        assert 'batch' not in keys
        # 创建空 batch 对象
        batch = BatchMasking()
        # 为每个字段创建空列表，用来存拼接数据
        for key in keys:
            batch[key] = []
        batch.batch = []
        cumsum_node = 0 # 节点编号偏移量
        cumsum_edge = 0 # 边编号偏移量
        # 遍历每个图
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes # 当前图节点数
            # 构造当前图节点所属图编号向量, i：当前图在 batch 中的索引（0, 1, 2 …）
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:  # 遍历当前图的每个字段
                item = data[key]
                # 如果是边索引或mask节点索引,需要加节点偏移量
                if key in ['edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                # 如果是连接边索引,需要加边偏移量
                elif key == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item) # 加入 batch 对应字段列表
            cumsum_node += num_nodes # 更新节点偏移量
            cumsum_edge += data.edge_index.shape[1]  # 更新边偏移量
        # 拼接所有图字段
        # Data 类定义了 __cat_dim__ 方法，用来告诉该key应该沿哪个维度拼接
        for key in keys:
            batch[key] = torch.cat(
                batch[key],
                dim=data_list[0].__cat_dim__(key, batch[key][0])
            )
        # batch 中总节点数 的 tensor,PyG 的全局操作（如 pooling）就是依赖这个索引来区分节点属于哪个图
        batch.batch = torch.cat(batch.batch, dim=-1) 

        return batch.contiguous() # 返回连续内存版本（提高计算性能）
    
    # 是否需要累计偏移的判断函数
    def cumsum(self, key, item):

        r"""如果返回 True，说明该字段需要在拼接前做偏移累加
        这个函数主要给内部机制调用
        """
        return key in [
            'edge_index',
            'face',
            'masked_atom_indices',
            'connected_edge_indices'
        ]
    
    @property
    def num_graphs(self):

        """返回当前 batch 中图的数量"""
        # batch[-1] 是最后一个节点所属图编号，+1 就是图总数
        return self.batch[-1].item() + 1

class BatchAE(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchAE, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchAE()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'negative_edge_index']:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.__cat_dim__(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "negative_edge_index"] else 0


class BatchSubstructContext(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Specialized batching for substructure context pair!
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        #keys = [set(data.keys) for data in data_list]
        #keys = list(set.union(*keys))
        #assert 'batch' not in keys

        batch = BatchSubstructContext()
        keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]

        for key in keys:
            #print(key)
            batch[key] = []

        #batch.batch = []
        #used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0
        
        for data in data_list:
            #If there is no context, just skip!!
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                #batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
                batch.batch_overlapped_context.append(torch.full((len(data.overlap_context_substruct_idx), ), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                ###batching for the main graph
                #for key in data.keys:
                #    if not "context" in key and not "substruct" in key:
                #        item = data[key]
                #        item = item + cumsum_main if batch.cumsum(key, item) else item
                #        batch[key].append(item)
                
                ###batching for the substructure graph
                for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)
                

                ###batching for the context graph
                for key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct   
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        #batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        return batch.contiguous()

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
