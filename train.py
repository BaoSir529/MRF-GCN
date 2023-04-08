import numpy as np
import torch
import torch.nn as nn
import argparse
import pandas as pd
import os
from utils.draw_utils import Draw
from model.mrf_model import Model
from utils.data_utils import load_data
import globalvar as gl
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

draw_list = gl.get_value('draw_list')
draw_list['time'] = ['2022/5/6', '2022/5/18', '2022/5/30', '2022/6/11', '2022/6/23', '2022/7/5', '2022/7/17', '2022/7/29', '2022/8/10', '2022/8/22', '2022/9/3', '2022/9/15', '2022/9/27', '2022/10/9', '2022/10/21', '2022/11/2', '2022/11/14', '2022/11/26', '2022/12/8', '2022/12/20']


def train(args, graph):
    train_loader, test_data = load_data(args)
    model = Model(args).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_func = nn.MSELoss()

    for epoch in range(args.epoch):
        model.train()
        for tx, ty in train_loader:
            inputs, labels = tx, ty   # inputs = batch * point_num * width
            output = model(inputs, graph)
            output = output.squeeze(-1)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch:{}, Loss:{}'.format(epoch, loss))

    print('Train end......')
    test(args, model, test_data, graph)


def test(args, model, data, graph):
    train_data_predict=[]
    test_data_predict=[]
    meta_data = data
    meta_data[:, 20:] = 0
    model.eval()
    for i in range(0, meta_data.size(1)-args.width):
        input = meta_data[:, i:i+args.width].unsqueeze(0)
        pred = model(input, graph).squeeze()

        if i < data.size(1) - args.test_num - args.width:
            train_data_predict.append(pred.tolist())
        else:
            test_data_predict.append(pred.tolist())
            meta_data[:, i+args.width] = pred

    pred = np.array(train_data_predict+test_data_predict).T
    for i in range(pred.shape[0]):
        draw_list['pred%d' %i] = pred[i:i+1, int(-args.test_num-10):].tolist()[0]
    if not os.path.exists("./output"):
        os.makedirs("./output")

    pd.DataFrame(draw_list).to_csv('./output/result.csv')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./data/input.csv', help='the path of the dataset.')
    parser.add_argument('--graph_path', type=str, default='./data/graph.csv', help='the path of the adjacent matrix.')
    parser.add_argument('--width', type=int, default=5, help='the width of the windows.')
    parser.add_argument('--lstm_hidden_dim', type=int, default=300, help='the hidden of the lstm.')
    parser.add_argument('--gcn_input_dim', type=int, default=300, help='the hidden of the gcn.')
    parser.add_argument('--gcn_hidden_dim', type=int, default=300, help='the output dim of the gcn.')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--train_data_num', type=int, default=89)

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.graph_path, 'rb') as f:
        graph = np.array(pd.read_csv(f, header=None))
    graph = torch.tensor(graph).float()
    graph = graph + torch.eye(graph.size(0))
    graph = graph.to(args.device)

    train(args, graph)
    Draw(draw_list)

