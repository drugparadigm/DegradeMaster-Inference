import numpy as np
import torch
import os
from flask import request
import logging
import json
from pathlib import  Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .protacloader import PROTACSet, collater
from .model import GraphConv, ProtacModel, GATTConv, EGNNConv
from .train_and_test import train, valids
from .config.config import get_args

from .prepare_data import GraphData
from .nn_utils import load_model

TRAIN_NAME = "test"
root = "src/data/input"
logging.basicConfig(filename="src/log/"+TRAIN_NAME+".log", filemode="w", level=logging.DEBUG)
args = get_args('case')

def main():
    
    reqId=request.form.get('reqId')

    with open(f"src/data/input/{reqId}_input.json", "r") as file:
        name_list_dic = json.load(file)

    name_dic = {}
    if len(name_list_dic) == 6:
        name_dic = {"c0": name_list_dic}
    
        
    name_list=list(name_dic.keys())
    protac_graphs = GraphData('protac', root=root,name_dic=name_dic,
                               select_pocket_war=args.select_pocket_war, select_pocket_e3=args.select_pocket_e3, conv_name=args.conv_name)
    ligase_pocket = GraphData("ligase_pocket", root=root,name_dic=name_dic,
                               select_pocket_war=args.select_pocket_war, select_pocket_e3=args.select_pocket_e3, conv_name=args.conv_name)
    target_pocket = GraphData("target_pocket", root=root,name_dic=name_dic,
                               select_pocket_war=args.select_pocket_war, select_pocket_e3=args.select_pocket_e3, conv_name=args.conv_name)

    label = torch.load(os.path.join(target_pocket.processed_dir, "label.pt",),weights_only=False)
    feature = torch.load(os.path.join(target_pocket.processed_dir, "feature.pt"),weights_only=False)

    protac_set = PROTACSet(
        name_list,
        protac_graphs,
        ligase_pocket,
        target_pocket,
        feature,
        label,
    )
    data_size = len(protac_set)

    pos_num, neg_num = 0, 0
    for key, value in name_dic.items():
        if value['label'] == 0:
            neg_num += 1
        elif value['label'] == 1:
            pos_num += 1
    logging.info(f"all data: {data_size}")
    logging.info(f"positive label number: {pos_num}")
    logging.info(f"negative label number: {neg_num}")


    testloader = DataLoader(protac_set, batch_size=args.batch_size, collate_fn=collater, drop_last=False,
                            shuffle=False)

    if args.conv_name == "GCN":
        ligase_pocket_model = GraphConv(num_embeddings=118, graph_dim=args.e3_dim, hidden_size=args.hidden_size)
        target_pocket_model = GraphConv(num_embeddings=118, graph_dim=args.tar_dim, hidden_size=args.hidden_size)
        protac_model = GraphConv(num_embeddings=118, graph_dim=args.protac_dim, hidden_size=args.hidden_size)
    elif args.conv_name == "GAT":
        ligase_pocket_model = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)
        target_pocket_model = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)
        protac_model = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)
    elif args.conv_name == "EGNN":
        ligase_pocket_model = EGNNConv(num_embeddings=118, in_node_nf=1, in_edge_nf=1,graph_nf=args.e3_dim, hidden_nf=args.hidden_size,
                                       n_layers=args.n_layers, node_attr=0, attention=args.attention)
        protac_model = EGNNConv(num_embeddings=118, in_node_nf=1, in_edge_nf=1, graph_nf=args.protac_dim, hidden_nf=args.hidden_size,
                                       n_layers=args.n_layers, node_attr=0, attention=args.attention)
        target_pocket_model = EGNNConv(num_embeddings=118, in_node_nf=1, in_edge_nf=1, graph_nf=args.tar_dim, hidden_nf=args.hidden_size,
                                       n_layers=args.n_layers, node_attr=0, attention=args.attention)
    else:
        raise ValueError("conv_type Error")

    model = ProtacModel(
        protac_model,
        ligase_pocket_model,
        target_pocket_model,
        args.hidden_size,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(f'src/runs/{TRAIN_NAME}')

    load_model(model, args, loaded_epoch=1000, case_study=True)
    if args.mode == 'Train':
        model = train(
            model,
            train_loader=testloader,
            valid_loader=testloader,
            device=device,
            writer=writer,
            LOSS_NAME=TRAIN_NAME,
            args=args
        )

    output_dic_test=valids(model.to(device),
                                test_loader=testloader,
                                device=device)
       

    return output_dic_test['pred'][0]

if __name__ == "__main__":
    main()