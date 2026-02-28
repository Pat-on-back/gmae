import argparse
from functools import partial # partial固定函数的一部分参数，返回新的该函数
from loader import MoleculeDataset
from dataloader import  DataLoaderMaskingPred 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from model import GNN, GNNDecoder


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def sce_loss(x, y, alpha=1):
    # x : [N, D] 预测向量
    # y : [N, D] 目标向量
    # alpha : 降低简单样本在训练中的贡献

    x = F.normalize(x, p=2, dim=-1) # 沿最后一维，使用欧几里得长度归一化
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()

    return loss


def train_mae(args, model_list, loader, optimizer_list, device, alpha_l=1.0, loss_fn="sce"):
    """
    Graph Masked AutoEncoder 单轮训练函数（训练一个epoch）

    参数说明：
    args                : 配置对象（包含是否mask边等参数）
    model_list          : 模型列表 [encoder, 节点decoder, 边decoder]
    loader              : 数据加载器（返回图batch）
    optimizer_list      : 优化器列表（分别对应三个模型）
    device              : 训练设备（cpu或cuda）
    alpha_l             : SCE损失指数参数
    loss_fn             : 损失函数类型 "sce" 或 "ce"
    """
    if loss_fn == "sce":
        # 创建了一个绑定 alpha=alpha_l 的
        # 新函数sce_loss(pred, target, alpha=alpha_l)
        criterion = partial(sce_loss, alpha=alpha_l)
    else:
        # 如果不是sce，则使用分类损失函数
        criterion = nn.CrossEntropyLoss()

    # model是encoder，ec_pred_atoms是节点预测decoder，
    # dec_pred_bonds是边预测decoder（可能为None）
    model, dec_pred_atoms, dec_pred_bonds = model_list
    optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds = optimizer_list
    
    model.train() # encoder启用训练模式
    dec_pred_atoms.train() # 节点decoder训练模式
    
    # 若存在边decoder，则也切换
    if dec_pred_bonds is not None:
        dec_pred_bonds.train()

    # 初始化统计变量
    loss_accum = 0         # 累计loss
    acc_node_accum = 0     # 节点准确率累计
    acc_edge_accum = 0     # 边准确率累计

    # 用进度条包装 DataLoader 迭代器
    epoch_iter = tqdm(loader, desc="Iteration")
    
    # 主训练循环
    for step, batch in enumerate(epoch_iter):
        batch = batch.to(device)

        #   batch.x            节点特征
        #   batch.edge_index   边索引
        #   batch.edge_attr    边属性
        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)

        # 节点真实标签（监督信号）
        node_attr_label = batch.node_attr_label
        
        # 被mask节点索引
        masked_node_indices = batch.masked_atom_indices

        pred_node = dec_pred_atoms(
            node_rep,
            batch.edge_index,
            batch.edge_attr,
            masked_node_indices
        )

        # ---------- 根据loss类型计算节点loss ----------
        if loss_fn == "sce":
            # SCE损失：比较向量方向
            # 只计算被mask节点
            loss = criterion(
                node_attr_label,
                pred_node[masked_node_indices]
            )
        else:
            # CrossEntropy分类损失
            # pred_node.double() -> 转为float64防止精度问题
            # mask_node_label[:,0] -> 从[M,1]转为[M]
            loss = criterion(
                pred_node.double()[masked_node_indices],
                batch.mask_node_label[:,0]
            )

        # 边mask重建任务（可选）
        if args.mask_edge:
            # 取出被mask的边索引
            masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            # 构造边表示，两端节点embedding相加
            edge_rep = (
                node_rep[masked_edge_index[0]] +
                node_rep[masked_edge_index[1]]
            )
            # 边decoder预测
            pred_edge = dec_pred_bonds(edge_rep)
            # 将边loss加入总loss
            loss += criterion(
                pred_edge.double(),
                batch.mask_edge_label[:,0]
            )
        optimizer_model.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()
        loss.backward()
        optimizer_model.step()               # 更新encoder参数
        optimizer_dec_pred_atoms.step()      # 更新节点decoder参数

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.step()  # 更新边decoder参数

        loss_accum += float(loss.cpu().item())
        # 更新进度条显示当前loss
        epoch_iter.set_description(
            f"train_loss: {loss.item():.4f}"
        )
    return loss_accum/step



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.25,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--input_model_file', type=str, default=None)
    parser.add_argument("--alpha_l", type=float, default=1.0)
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--decoder", type=str, default="gin")
    parser.add_argument("--use_scheduler", action="store_true", default=False)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" %(args.num_layer, args.mask_rate, args.mask_edge))


    dataset_name = args.dataset
    dataset = MoleculeDataset("dataset/" + dataset_name, dataset=dataset_name)
    loader = DataLoaderMaskingPred(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, mask_rate=args.mask_rate, mask_edge=args.mask_edge)

    # set up models, one for pre-training and one for context embeddings
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)
    
    if args.input_model_file is not None and args.input_model_file != "":
        model.load_state_dict(torch.load(args.input_model_file))
        print("Resume training from:", args.input_model_file)
        resume = True
    else:
        resume = False

    NUM_NODE_ATTR = 119 # + 3 
    atom_pred_decoder = GNNDecoder(args.emb_dim, NUM_NODE_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(device)
    if args.mask_edge:
        NUM_BOND_ATTR = 5 + 3
        bond_pred_decoder = GNNDecoder(args.emb_dim, NUM_BOND_ATTR, JK=args.JK, gnn_type=args.gnn_type)
        optimizer_dec_pred_bonds = optim.Adam(bond_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        bond_pred_decoder = None
        optimizer_dec_pred_bonds = None

    model_list = [model, atom_pred_decoder, bond_pred_decoder] 

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.use_scheduler:
        print("--------- Use scheduler -----------")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.epochs) ) * 0.5
        scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=scheduler)
        scheduler_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms, lr_lambda=scheduler)
        scheduler_list = [scheduler_model, scheduler_dec, None]
    else:
        scheduler_model = None
        scheduler_dec = None

    optimizer_list = [optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds]

    output_file_temp = "./checkpoints/" + args.output_model_file + f"_{args.gnn_type}"

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        # train_loss, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, device)
        # print(train_loss, train_acc_atom, train_acc_bond)

        train_loss = train_mae(args, model_list, loader, optimizer_list, device, alpha_l=args.alpha_l, loss_fn=args.loss_fn)
        if not resume:
            if epoch % 50 == 0:
                torch.save(model.state_dict(), output_file_temp + f"_{epoch}.pth")
        print(train_loss)
        if scheduler_model is not None:
            scheduler_model.step()
        if scheduler_dec is not None:
            scheduler_dec.step()

    output_file = "./checkpoints/" + args.output_model_file + f"_{args.gnn_type}"
    if resume:
        torch.save(model.state_dict(), args.input_model_file.rsplit(".", 1)[0] + f"_resume_{args.epochs}.pth")
    elif not args.output_model_file == "":
        torch.save(model.state_dict(), output_file + ".pth")

if __name__ == "__main__":
    main()
