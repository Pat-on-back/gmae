import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class GINConv(MessagePassing):  
    # 定义一个图卷积层类 GINConv
    # 继承自 PyTorch Geometric 的 MessagePassing 基类
    # 该基类封装了标准图神经网络传播流程：
    # message → aggregate → update
    """
    GIN卷积层的扩展版本：在原始GIN基础上加入边特征信息（edge feature）

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        emb_dim：节点特征向量维度，同时也是边embedding维度
    """

    def __init__(self, emb_dim, out_dim, aggr="add", **kwargs):
        # emb_dim  → 输入节点特征维度
        # out_dim  → 输出节点特征维度
        # aggr     → 聚合方式（add/mean/max）
        # kwargs   → 传给父类 MessagePassing 的额外参数

        kwargs.setdefault('aggr', aggr)
        # 如果kwargs中没有aggr字段
        # 就设置为当前传入的aggr
        # 目的是让父类MessagePassing知道使用什么聚合方式
        self.aggr = aggr
        super(GINConv, self).__init__(**kwargs)
        # 调用父类构造函数
        # 初始化MessagePassing内部机制
        # 包括：message参数解析，edge索引映射规则，scatter聚合方式绑定

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, out_dim)
        
        )
        # 边特征Embedding层，输入：键类型ID（整数），输出：emb_dim维向量
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        # 边方向 embedding  输入：方向ID，输出：emb_dim向量
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        # 初始化embedding权重（Xavier初始化）
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        # Xavier初始化可以保持前向传播与反向传播的方差稳定，避免梯度爆炸或消失

        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        """
        x           节点特征矩阵        shape = [N , emb_dim]
        edge_index  边索引矩阵          shape = [2 , E]
        edge_attr   边属性矩阵          shape = [E , 2]
        """
        # Step1：为图添加自环边 (self loop)
        # h_i' = MLP((1+ε)h_i + Σ邻居h_j)，加自环相当于让节点自己成为邻居
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step2：为自环边构造边特征
        # 创建 shape=[N,2] 的矩阵
        # 每一行对应一条自环边的特征
        self_loop_attr = torch.zeros(x.size(0), 2)

        # 设置第0列为4
        # 表示：bond_type = 4 代表“自环类型”
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        # 将自环边特征拼接到原边特征末尾
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
       
        # Step3：将离散边特征映射为向量（Embedding）
        # edge_attr[:,0] → 键类型ID
        # edge_attr[:,1] → 键方向ID
        edge_embeddings = (
            self.edge_embedding1(edge_attr[:, 0]) +
            self.edge_embedding2(edge_attr[:, 1])
        )
        
        # Step4：调用消息传播机制
        # propagate 是 MessagePassing 核心函数
        # 自动执行：1. message() 2. aggregate() 3. update()
        # 并根据 edge_index 自动匹配邻居关系
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        
    # message函数 —— 定义边上传递的信息
    def message(self, x_j, edge_attr):
        # x_j 表示邻居节点特征
        # edge_attr 表示该边特征
        # 定义消息公式：m_{j→i} = x_j + e_{ji}
        # 即：邻居节点信息 + 边信息
        return x_j + edge_attr
       
    # update函数 —— 节点更新规则
    def update(self, aggr_out):
        # 节点更新公式：h_i' = MLP( Σ_j (x_j + e_{ji}) )
        return self.mlp(aggr_out)

class GCNConv(MessagePassing):

    def __init__(self, emb_dim, out_dim, aggr = "add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GCNConv, self).__init__(**kwargs)

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, out_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        # x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)
        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)
    
    # added
    def update(self, aggr_out):
        return self.linear(aggr_out)



class GATConv(MessagePassing):
    def __init__(self, emb_dim, out_dim, heads=2, negative_slope=0.2, aggr = "add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)



class GNN(torch.nn.Module):
    """
    图神经网络主模型类
    作用：根据输入图结构与节点特征，计算每个节点的表示向量

    参数说明:
        num_layer (int): GNN层数（必须 ≥2）
        emb_dim (int): 节点embedding维度
        JK (str): Jumping Knowledge策略
                  可选: "last", "concat", "max", "sum"
        drop_ratio (float): dropout概率
        gnn_type (str): GNN层类型
                        可选: gin / gcn / gat / graphsage

    输出:
        node representations (每个节点的向量表示)
    """

    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        # 调用父类构造函数（必须写，否则Module机制失效）
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        # 至少需要2层GNN，否则模型表达能力不足
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # ========== 节点离散特征 embedding ==========
        # x[:,0] = 原子类型
        # embedding矩阵大小: [num_atom_type, emb_dim]
        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)

        # x[:,1] = 手性标签
        # embedding矩阵大小: [num_chirality_tag, emb_dim]
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        # 使用Xavier均匀分布初始化权重（深度学习常用初始化）
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.gnns = torch.nn.ModuleList()

        for layer in range(num_layer):

            # 不同类型图卷积
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        # 每一层GNN后都加一个BN稳定训练
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            # BN作用维度 = embedding维度
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            # 直接传入张量
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]

        elif len(argv) == 1:
            # 传入PyG data对象
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        else:
            # 参数数量不匹配直接报错
            raise ValueError("unmatched number of arguments.")

        # ========== 节点embedding ==========
        # x形状: [num_nodes, 2]
        # x[:,0] 原子类型index
        # x[:,1] 手性index，分别embedding后相加融合
        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        # 输出维度: [num_nodes, emb_dim]
        # h_list保存每一层输出（用于JK）
        h_list = [x]
        # ========== GNN层迭代 ==========
        for layer in range(self.num_layer):
            # 当前层GNN计算
            # 输入: 上一层节点特征 + 图结构
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            # BN归一化（稳定梯度）
            h = self.batch_norms[layer](h)
            # 最后一层特殊处理
            if layer == self.num_layer - 1:
                # 最后一层不使用ReLU，需要保留完整表达能力
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                # 中间层：ReLU + Dropout
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        # ========== Jumping Knowledge策略 ==========
        # 不同策略融合多层特征
        if self.JK == "concat":
            # 拼接所有层特征
            # 维度: [num_nodes, emb_dim * (num_layer+1)]
            node_representation = torch.cat(h_list, dim = 1)

        elif self.JK == "last":
            # 只用最后一层
            node_representation = h_list[-1]

        elif self.JK == "max":
            # 逐层最大值融合
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            # 多层特征逐元素相加
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]
        # 返回节点表示
        return node_representation

class GNNDecoder(torch.nn.Module):
    # 定义图神经网络解码器模块
    # 根据encoder输出的节点表示，预测节点属性或类别（常用于mask重建、自监督任务）
    def __init__(self, hidden_dim, out_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        # hidden_dim : 输入节点特征维度（encoder输出维度）
        # out_dim    : 输出维度（预测维度/类别数）
        # JK         : 保留参数（未使用，为兼容接口）
        # drop_ratio : 保留参数（未使用，为兼容接口）
        # gnn_type   : 解码器类型，可选 gin/gcn/linear
        super().__init__()  # 初始化父类Module（必须）
        self._dec_type = gnn_type 

        # ===== 根据类型选择解码结构 =====
        if gnn_type == "gin":
            self.conv = GINConv(hidden_dim, out_dim, aggr = "add")
        elif gnn_type == "gcn":
            self.conv = GCNConv(hidden_dim, out_dim, aggr = "add")

        elif gnn_type == "linear":
            # 如果选择linear解码器，则完全不使用图结构，直接做逐节点线性变换
            self.dec = torch.nn.Linear(hidden_dim, out_dim)
        else:
            raise NotImplementedError(f"{gnn_type}")

        # ===== mask token 向量 =====
        # 定义一个可学习参数，用作mask节点替代向量
        # shape = [1, hidden_dim]
        # 类似BERT中的[MASK] embedding
        self.dec_token = torch.nn.Parameter(torch.zeros([1, hidden_dim]))

        # ===== encoder → decoder 映射层 =====
        # 用于将encoder输出空间映射到decoder空间
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)    

        # PReLU是可学习负斜率ReLU
        # 优点：模型能自动学习负区间斜率
        self.activation = torch.nn.PReLU() 

        # temp越小 → 概率分布越尖锐
        self.temp = 0.2


    def forward(self, x, edge_index, edge_attr, mask_node_indices):
        """
        前向传播函数

        参数说明:
        x : [N, hidden_dim]   encoder输出节点特征
        edge_index : [2, E]   图结构边索引
        edge_attr  : [E, ?]   边特征
        mask_node_indices : 被mask节点索引列表
        返回:
        out : [N, out_dim]    每个节点预测结果
        """
        # ===== 如果是线性decoder =====
        if self._dec_type == "linear":
            # 直接线性映射，不使用图结构
            out = self.dec(x)
        # ===== 如果是图卷积decoder =====
        else:
            # 先做激活函数，提供非线性能力
            x = self.activation(x)

            # encoder表示 → decoder表示空间映射
            x = self.enc_to_dec(x)

            # 将mask节点特征清零，禁止模型使用自身信息
            # 强制模型利用邻居信息恢复该节点
            x[mask_node_indices] = 0

            # 图卷积预测输出，利用邻居节点信息进行推理
            out = self.conv(x, edge_index, edge_attr)
            
        # 返回节点预测结果
        return out


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


if __name__ == "__main__":
    pass

