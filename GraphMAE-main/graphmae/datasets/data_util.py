
from collections import namedtuple, Counter
import numpy as np

import torch
import torch.nn.functional as F

import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader

from sklearn.preprocessing import StandardScaler


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset
}


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats




def load_dataset(dataset_name):
    # 断言：数据集名称必须存在于预定义数据集字典 GRAPH_DICT 中，否则报错
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."

    # 若数据集名以 "ogbn" 开头（OGB节点分类基准数据集）
    if dataset_name.startswith("ogbn"):
        # OGB数据集初始化时需要传入数据集名称参数
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        # 普通数据集直接实例化即可
        dataset = GRAPH_DICT[dataset_name]()
    
    # 特殊处理：ogbn-arxiv 数据集
    if dataset_name == "ogbn-arxiv":
        # dataset[0] 返回一个 tuple: (图对象, 标签张量)
        graph, labels = dataset[0]
        # 获取节点总数
        num_nodes = graph.num_nodes()
        # 获取官方划分的训练/验证/测试索引
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        # 对图进行预处理（如标准化邻接矩阵、添加反向边等）
        graph = preprocess(graph)
        # 若索引不是 Tensor 类型，则转为 Tensor
        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)
        # 读取节点特征矩阵 (N × F)
        feat = graph.ndata["feat"]
        # 对特征做归一化/标准化处理sklearn StandardScaler
        feat = scale_feats(feat)
        # 将处理后的特征重新写回图数据
        graph.ndata["feat"] = feat
        # 构造训练 mask（长度=节点数，全 False，再把训练索引位置设为 True）
        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        # 构造验证 mask
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        # 构造测试 mask
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        # 写入标签（展平成一维）
        graph.ndata["label"] = labels.view(-1)
        # 将三类 mask 写入图节点属性字典
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask

    # 普通数据集处理流程
    else:
        # dataset[0] 直接返回图对象
        graph = dataset[0]
        # 删除已有自环（避免重复）
        graph = graph.remove_self_loop()
        # 添加标准自环（GCN等模型常要求）
        graph = graph.add_self_loop()

    # 获取特征维度 F
    num_features = graph.ndata["feat"].shape[1]
    # 获取类别数 C
    num_classes = dataset.num_classes
    # 返回：图对象 + (特征维度, 类别数)
    return graph, (num_features, num_classes)


def load_inductive_dataset(dataset_name):
    if dataset_name == "ppi":
        batch_size = 2
        # define loss function
        # create the dataset
        train_dataset = PPIDataset(mode='train')
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')
        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]
    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]
        
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes



def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
            
            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)
