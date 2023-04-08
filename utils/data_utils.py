import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import globalvar as gl

draw_list = gl.get_value('draw_list')


class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :, :-1].float(), data[:, :, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def data(args):
    with open(args.data_path) as f:
        point_data = np.array(pd.read_csv(f, header=None))
    for i in range(point_data.shape[0]):
        draw_list['meta%d' % i] = point_data[i:i + 1, int(-args.test_num - 10):].tolist()[0]
    return point_data


def load_data(args):
    width = args.width
    point_data = data(args)
    train_data = []
    for point in range(point_data.shape[0]):
        per_point_data = pd.Series(point_data[point][:-args.test_num])
        if len(per_point_data) <= width:
            raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(per_point_data), width))
        df = pd.DataFrame()
        for i in range(width):
            df['step%d' % i] = per_point_data.tolist()[i:-(width - i)]
        df['y'] = per_point_data.tolist()[width:]
        df.index = per_point_data.index[width:]

        df_numpy = np.array(df)
        train_data.append(df_numpy.tolist())

    train_data = torch.tensor(train_data).to(args.device)
    test_data = torch.tensor(point_data[:, -args.test_num - 20:]).float().to(args.device)

    train_set = TrainSet(train_data.permute(1, 0, 2))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_data
