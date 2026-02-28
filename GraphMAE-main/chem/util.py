import torch
import torch.nn.functional as F

import copy
import random
import networkx as nx
import numpy as np
from torch_geometric.utils import convert
from loader import graph_data_obj_to_nx_simple, nx_to_graph_data_obj_simple
from rdkit import Chem
from rdkit.Chem import AllChem
from loader import mol_to_graph_data_obj_simple, \
    graph_data_obj_to_mol_simple

from loader import MoleculeDataset


def check_same_molecules(s1, s2):
    mol1 = AllChem.MolFromSmiles(s1)
    mol2 = AllChem.MolFromSmiles(s2)
    return AllChem.MolToInchi(mol1) == AllChem.MolToInchi(mol2)


class NegativeEdge:
    def __init__(self):
        """
        Randomly sample negative edges
        """
        pass

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        edge_set = set([str(data.edge_index[0, i].cpu().item()) + "," + str(
            data.edge_index[1, i].cpu().item()) for i in
                        range(data.edge_index.shape[1])])

        redandunt_sample = torch.randint(0, num_nodes, (2, 5 * num_edges))
        sampled_ind = []
        sampled_edge_set = set([])
        for i in range(5 * num_edges):
            node1 = redandunt_sample[0, i].cpu().item()
            node2 = redandunt_sample[1, i].cpu().item()
            edge_str = str(node1) + "," + str(node2)
            if not edge_str in edge_set and not edge_str in sampled_edge_set and not node1 == node2:
                sampled_edge_set.add(edge_str)
                sampled_ind.append(i)
            if len(sampled_ind) == num_edges / 2:
                break

        data.negative_edge_index = redandunt_sample[:, sampled_ind]

        return data


class ExtractSubstructureContextPair:
    def __init__(self, k, l1, l2):
        """
        Randomly selects a node from the data object, and adds attributes
        that contain the substructure that corresponds to k hop neighbours
        rooted at the node, and the context substructures that corresponds to
        the subgraph that is between l1 and l2 hops away from the
        root node.
        :param k:
        :param l1:
        :param l2:
        """
        self.k = k
        self.l1 = l1
        self.l2 = l2

        # for the special case of 0, addresses the quirk with
        # single_source_shortest_path_length
        if self.k == 0:
            self.k = -1
        if self.l1 == 0:
            self.l1 = -1
        if self.l2 == 0:
            self.l2 = -1

    def __call__(self, data, root_idx=None):
        """

        :param data: pytorch geometric data object
        :param root_idx: If None, then randomly samples an atom idx.
        Otherwise sets atom idx of root (for debugging only)
        :return: None. Creates new attributes in original data object:
        data.center_substruct_idx
        data.x_substruct
        data.edge_attr_substruct
        data.edge_index_substruct
        data.x_context
        data.edge_attr_context
        data.edge_index_context
        data.overlap_context_substruct_idx
        """
        num_atoms = data.x.size()[0]
        if root_idx == None:
            root_idx = random.sample(range(num_atoms), 1)[0]

        G = graph_data_obj_to_nx_simple(data)  # same ordering as input data obj

        # Get k-hop subgraph rooted at specified atom idx
        substruct_node_idxes = nx.single_source_shortest_path_length(G,
                                                                     root_idx,
                                                                     self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
            # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            substruct_data = nx_to_graph_data_obj_simple(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_attr_substruct = substruct_data.edge_attr
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[
                                                          root_idx]])  # need
            # to convert center idx from original graph node ordering to the
            # new substruct node ordering

        # Get subgraphs that is between l1 and l2 hops away from the root node
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            context_data = nx_to_graph_data_obj_simple(context_G)
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(set(
            context_node_idxes).intersection(set(substruct_node_idxes)))
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for
                                                       old_idx in
                                                       context_substruct_overlap_idxes]
            # need to convert the overlap node idxes, which is from the
            # original graph node ordering to the new context node ordering
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)

        return data

        # ### For debugging ###
        # if len(substruct_node_idxes) > 0:
        #     substruct_mol = graph_data_obj_to_mol_simple(data.x_substruct,
        #                                                  data.edge_index_substruct,
        #                                                  data.edge_attr_substruct)
        #     print(AllChem.MolToSmiles(substruct_mol))
        # if len(context_node_idxes) > 0:
        #     context_mol = graph_data_obj_to_mol_simple(data.x_context,
        #                                                data.edge_index_context,
        #                                                data.edge_attr_context)
        #     print(AllChem.MolToSmiles(context_mol))
        #
        # print(list(context_node_idxes))
        # print(list(substruct_node_idxes))
        # print(context_substruct_overlap_idxes)
        # ### End debugging ###

    def __repr__(self):
        return '{}(k={},l1={}, l2={})'.format(self.__class__.__name__, self.k,
                                              self.l1, self.l2)


def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        随机掩码一个原子，并可选择性地掩码与该原子相连的边。
        被掩码的原子类型索引设为 num_possible_atom_type。
        被掩码的边类型索引设为 num_possible_edge_type。
        参数：
            num_atom_type: 可能的原子类型数量（用于设定掩码索引）。
            num_edge_type: 可能的边类型数量（用于设定掩码索引）。
            mask_rate: 需要被掩码的原子百分比。
            mask_edge: 如果为 True，则同时掩码连接到被掩码原子的边。
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.num_chirality_tag = 3 
        self.num_bond_direction = 3 

    # 调用入口，使对象可像函数一样调用
    def __call__(self, data, masked_atom_indices=None):
        """
        参数：
            data: PyTorch Geometric 数据对象。假设边的排序遵循 PyTorch Geometric 的默认顺序，
            即单条边的两个方向成对出现。例如：
            data.edge_index = tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
            masked_atom_indices: 如果为 None，则随机采样 原子总数 * mask_rate 数量的原子索引。
            否则，使用提供的原子索引列表来指定要掩码的原子（仅用于调试）。

        返回：None。该函数会在原始数据对象中创建以下新属性：
            data.mask_node_idx：被掩码的节点（原子）索引。
            data.mask_node_label：被掩码节点的原始标签（用于后续预测任务）。
            data.mask_edge_idx：被掩码的边索引。
            data.mask_edge_label：被掩码边的原始标签。
        """
        # 如果没有指定 mask 节点，就随机采样。
        
        if masked_atom_indices == None:
            num_atoms = data.x.size()[0] # 计算节点总数
            sample_size = int(num_atoms * self.mask_rate + 1) # 计算采样数
            # 无重复、均匀、无放回采样sample_size个
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        mask_node_labels_list = [] # 保存节点特征（监督信号）
        for atom_idx in masked_atom_indices:
            # view(1,-1) 变为[1, feature_dim]方便拼接
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1)) 
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0) # 保存mask节点特征
        data.masked_atom_indices = torch.tensor(masked_atom_indices) # 保存mask索引

        # ----------- graphMAE -----------
        # one-hot 向量，shape：[num_masked_nodes, num_atom_type]
        # data.mask_node_label[:, 0]取 mask 节点的 原子类型索引，如 [5, 0, 6]代表 C、H、N 等原子
        """
        假设 self.num_atom_type = 6，输入 [1,3] → 输出：
        [[0,1,0,0,0,0],
        [0,0,0,1,0,0]]
        """
        atom_type = F.one_hot(data.mask_node_label[:, 0], num_classes=self.num_atom_type).float()
        # 最终标节点标签
        data.node_attr_label = atom_type

        # 修改原节点特征以进行 mask
        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            # 找出与被 mask 节点相连的所有边，保存所有相关边索引
            connected_edge_indices = []
            # bond_idx 是边的索引， (u, v) 是边的起点和终点原子编号
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    # 若当前边的两个节点中包含 mask 原子，并避免重复添加
                    if atom_idx in set((u, v)) and bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # 复制边特征作为标签
                mask_edge_labels_list = []
                # [::2]切片操作，每隔一个取一个元素
                for bond_idx in connected_edge_indices[::2]: # 因为边的顺序是这样的：单条边的两个方向成对出现，
                    # 所以为了获取唯一的无向边索引，我们从列表中每隔一个边索引取一个。
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # mask边特征
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])
                    
                # 仅保存去重后的边索引。
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
                
            # 如果没有与 mask 节点相连的边，则生成空 tensor。
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)
            # edge one-hot 编码
            edge_type = F.one_hot(data.mask_edge_label[:, 0], num_classes=self.num_edge_type).float()
            bond_direction = F.one_hot(data.mask_edge_label[:, 1], num_classes=self.num_bond_direction).float()
            data.edge_attr_label = torch.cat((edge_type, bond_direction), dim=1)
            # data.edge_attr_label = edge_type

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


if __name__ == "__main__":
    transform = NegativeEdge()
    dataset = MoleculeDataset("dataset/tox21", dataset="tox21")
    transform(dataset[0])

    """
    # TODO(Bowen): more unit tests
    # test ExtractSubstructureContextPair

    smiles = 'C#Cc1c(O)c(Cl)cc(/C=C/N)c1S'
    m = AllChem.MolFromSmiles(smiles)
    data = mol_to_graph_data_obj_simple(m)
    root_idx = 13

    # 0 hops: no substructure or context. We just test the absence of x attr
    transform = ExtractSubstructureContextPair(0, 0, 0)
    transform(data, root_idx)
    assert not hasattr(data, 'x_substruct')
    assert not hasattr(data, 'x_context')

    # k > n_nodes, l1 = 0 and l2 > n_nodes: substructure and context same as
    # molecule
    data = mol_to_graph_data_obj_simple(m)
    transform = ExtractSubstructureContextPair(100000, 0, 100000)
    transform(data, root_idx)
    substruct_mol = graph_data_obj_to_mol_simple(data.x_substruct,
                                                 data.edge_index_substruct,
                                                 data.edge_attr_substruct)
    context_mol = graph_data_obj_to_mol_simple(data.x_context,
                                               data.edge_index_context,
                                               data.edge_attr_context)
    assert check_same_molecules(AllChem.MolToSmiles(substruct_mol),
                                AllChem.MolToSmiles(context_mol))

    transform = ExtractSubstructureContextPair(1, 1, 10000)
    transform(data, root_idx)

    # increase k from 0, and increase l1 from 1 while keeping l2 > n_nodes: the
    # total number of atoms should be n_atoms
    for i in range(len(m.GetAtoms())):
        data = mol_to_graph_data_obj_simple(m)
        print('i: {}'.format(i))
        transform = ExtractSubstructureContextPair(i, i, 100000)
        transform(data, root_idx)
        if hasattr(data, 'x_substruct'):
            n_substruct_atoms = data.x_substruct.size()[0]
        else:
            n_substruct_atoms = 0
        print('n_substruct_atoms: {}'.format(n_substruct_atoms))
        if hasattr(data, 'x_context'):
            n_context_atoms = data.x_context.size()[0]
        else:
            n_context_atoms = 0
        print('n_context_atoms: {}'.format(n_context_atoms))
        assert n_substruct_atoms + n_context_atoms == len(m.GetAtoms())

    # l1 < k and l2 >= k, so an overlap exists between context and substruct
    data = mol_to_graph_data_obj_simple(m)
    transform = ExtractSubstructureContextPair(2, 1, 3)
    transform(data, root_idx)
    assert hasattr(data, 'center_substruct_idx')

    # check correct overlap atoms between context and substruct


    # m = AllChem.MolFromSmiles('COC1=CC2=C(NC(=N2)[S@@](=O)CC2=NC=C(C)C(OC)=C2C)C=C1')
    # data = mol_to_graph_data_obj_simple(m)
    # root_idx = 9
    # k = 1
    # l1 = 1
    # l2 = 2
    # transform = ExtractSubstructureContextPaidata = mol_to_graph_data_obj_simple(m)r(k, l1, l2)
    # transform(data, root_idx)
    pass

    # TODO(Bowen): more unit tests
    # test MaskAtom
    from loader import mol_to_graph_data_obj_simple, \
        graph_data_obj_to_mol_simple

    smiles = 'C#Cc1c(O)c(Cl)cc(/C=C/N)c1S'
    m = AllChem.MolFromSmiles(smiles)
    original_data = mol_to_graph_data_obj_simple(m)
    num_atom_type = 118
    num_edge_type = 5

    # manually specify masked atom indices, don't mask edge
    masked_atom_indices = [13, 12]
    data = mol_to_graph_data_obj_simple(m)
    transform = MaskAtom(num_atom_type, num_edge_type, 0.1, mask_edge=False)
    transform(data, masked_atom_indices)
    assert data.mask_node_label.size() == torch.Size(
        (len(masked_atom_indices), 2))
    assert not hasattr(data, 'mask_edge_label')
    # check that the correct rows in x have been modified to be mask atom type
    assert (data.x[masked_atom_indices] == torch.tensor(([num_atom_type,
                                                          0]))).all()
    assert (data.mask_node_label == original_data.x[masked_atom_indices]).all()

    # manually specify masked atom indices, mask edge
    masked_atom_indices = [13, 12]
    data = mol_to_graph_data_obj_simple(m)
    transform = MaskAtom(num_atom_type, num_edge_type, 0.1, mask_edge=True)
    transform(data, masked_atom_indices)
    assert data.mask_node_label.size() == torch.Size(
        (len(masked_atom_indices), 2))
    # check that the correct rows in x have been modified to be mask atom type
    assert (data.x[masked_atom_indices] == torch.tensor(([num_atom_type,
                                                          0]))).all()
    assert (data.mask_node_label == original_data.x[masked_atom_indices]).all()
    # check that the correct rows in edge_attr have been modified to be mask edge
    # type, and the mask_edge_label are correct
    rdkit_bonds = []
    for atom_idx in masked_atom_indices:
        bond_indices = list(AllChem.FindAtomEnvironmentOfRadiusN(m, radius=1,
                                                                 rootedAtAtom=atom_idx))
        for bond_idx in bond_indices:
            rdkit_bonds.append(
                (m.GetBonds()[bond_idx].GetBeginAtomIdx(), m.GetBonds()[
                    bond_idx].GetEndAtomIdx()))
            rdkit_bonds.append(
                (m.GetBonds()[bond_idx].GetEndAtomIdx(), m.GetBonds()[
                    bond_idx].GetBeginAtomIdx()))
    rdkit_bonds = set(rdkit_bonds)
    connected_edge_indices = []
    for i in range(data.edge_index.size()[1]):
        if tuple(data.edge_index.numpy().T[i].tolist()) in rdkit_bonds:
            connected_edge_indices.append(i)
    assert (data.edge_attr[connected_edge_indices] ==
            torch.tensor(([num_edge_type, 0]))).all()
    assert (data.mask_edge_label == original_data.edge_attr[
        connected_edge_indices[::2]]).all() # data.mask_edge_label contains
    # the unique edges (ignoring direction). The data obj has edge ordering
    # such that two directions of a single edge occur in pairs, so to get the
    # unique undirected edge indices, we take every 2nd edge index from list
    """

