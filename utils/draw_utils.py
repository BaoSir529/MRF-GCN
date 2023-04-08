import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def Draw(draw_list):
    plt.style.use('ggplot')
    plt.figure(figsize=(18, 13.5))
    plt.plot(draw_list['time'], draw_list['meta19'],'-o',label='Original', color='red',  markeredgecolor='r', markerfacecolor='w', markersize=15)
    plt.plot(draw_list['time'], draw_list['pred19'],'--D',label='Prediction', color='b', markeredgecolor='b', markerfacecolor='y', markersize=15)
    plt.xlabel("Date", loc='right', fontsize=35)
    plt.ylabel("Elevation/mm", loc='top', fontsize=35)
    plt.xticks(rotation=25)
    plt.tick_params(labelsize=20)
    plt.title("Time series image of point p19", fontsize=50)
    plt.grid(True)
    plt.legend(loc='lower right', prop = {'size':30})
    plt.savefig('./output/insar_p19.eps')
    plt.show()


def line_compre(draw_list, lstm_data):
    plt.style.use('_classic_test_patch')
    plt.figure(figsize=(18, 13.5))
    plt.plot(draw_list['time'], draw_list['meta7'], '-o', label='Original', color='red', markeredgecolor='r',
             markerfacecolor='w', markersize=15)
    plt.plot(draw_list['time'], draw_list['pred7'], '--D', label='MRF-GCN', color='g', markeredgecolor='g',
             markerfacecolor='y', markersize=15)
    plt.plot(draw_list['time'], lstm_data['pred7'], '-.^', label='LSTM', color='b', markeredgecolor='b',
             markerfacecolor='y', markersize=15)
    plt.xlabel("Date", loc='right', fontsize=30)
    plt.ylabel("Elevation/mm", loc='top', fontsize=30)
    plt.xticks(rotation=25)
    plt.tick_params(labelsize=20)
    plt.title("Manual Level Measurement", fontsize=30)
    plt.grid(True)
    plt.legend(loc='lower right', prop={'size': 30})
    plt.savefig('./output/human_gcn_lstm_p7.eps')
    plt.show()


def scarter(paper_data, lstm_data, arima_data):
    plt.style.use('fast')
    plt.figure(figsize=(10, 10))
    plt.plot(paper_data['meta19'], paper_data['meta19'], '-' ,color='black', marker='+', markersize=8, zorder=1, label='Y=X')
    plt.plot(paper_data['meta19'], paper_data['meta19']+3, '-', color='gray', marker='+', markersize=8, zorder=1, label='Y=X±3')
    plt.plot(paper_data['meta19'], paper_data['meta19']-3,'-' , color='gray', marker='+', markersize=8, zorder=1)

    plt.scatter(paper_data['meta19'], paper_data['pred19'], color='r', marker='D', s=100, zorder=3, label='Our model')
    plt.scatter(lstm_data['meta19'], lstm_data['pred19'], color='g', marker='o', s=100, zorder=2, label='LSTM')
    plt.scatter(arima_data['meta19'], arima_data['pred19'], color='b', marker='^', s=100, zorder=2, label='ARIMA')
    # plt.plot(draw_list['time'], draw_list['pred19'], '--D', label='Prediction', color='b', markeredgecolor='b',markerfacecolor='y', markersize=8)
    plt.xlabel("Real values", fontsize=35)
    plt.ylabel("Prediction values", fontsize=35)
    plt.xlim(15,32)
    plt.ylim(15,32)
    plt.tick_params(labelsize=20)
    plt.grid(False)
    plt.legend(loc='upper left',prop = {'size':20})
    plt.savefig('./output/error_chart.eps')
    plt.show()


def draw_heatmap(gcn_list, lstm_list):
    # mask
    with open('./graph.csv','rb') as f:
        mask = np.array(pd.read_csv(f, header=None))
    mask = torch.tensor(mask).float()
    # mask = mask + torch.eye(mask.size(0))
    mask = mask.numpy()

    plt.figure(figsize=(23, 10))
    plt.subplot(1,2,1)
    gcn_map = torch.zeros(20, 20)
    for i in range(20):
        vec1 = torch.FloatTensor(gcn_list[f'pred{i}'].tolist())
        for j in range(20):
            vec2 = torch.FloatTensor(gcn_list[f'pred{j}'].tolist())
    #         # map[i][j] = F.cosine_similarity(vec1, vec2, dim=0)
    #         # map[i][j] = r2_score(vec1, vec2)
            gcn_map[i][j] = np.corrcoef(vec1, vec2)[0][1]
    # map = torch.sigmoid(lstm_map)
    sns.heatmap(gcn_map,  vmin=-1, vmax=1, cmap="hot_r", linewidths=0.3, mask=mask<1,linecolor="grey")
    plt.title('MRF-GCN', fontsize=18)

    plt.subplot(1,2,2)
    lstm_map = torch.zeros(20, 20)
    for i in range(20):
        vec1 = torch.FloatTensor(lstm_list[f'pred{i}'].tolist())
        for j in range(20):
            vec2 = torch.FloatTensor(lstm_list[f'pred{j}'].tolist())
    #         # lstm_map[i][j] = F.cosine_similarity(vec1, vec2, dim=0)
    #         # lstm_map[i][j] = r2_score(vec1, vec2)
            lstm_map[i][j] = np.corrcoef(vec1, vec2)[0][1]
    # lstm_map = torch.sigmoid(lstm_map)
    sns.heatmap(lstm_map, vmin=-1, vmax=1, cmap="hot_r", linewidths=0.3, mask=mask<1,linecolor="grey")
    plt.title('LSTM', fontsize=18)

    plt.savefig('./heatmap.eps', bbox_inches='tight')
    plt.show()

    return 0


if __name__ == '__main__':
    with open('./output/result.csv') as f:
        draw_list = pd.read_csv(f)

    with open('./output/lstm_result.csv') as f:
        lstm_data = pd.read_csv(f)

    with open('./output/arima_result.csv') as f:
        arima_data = pd.read_csv(f)
    # Draw(draw_list)
    # line_compre(draw_list, lstm_data)
    # scarter(draw_list, lstm_data, arima_data)
    # draw_heatmap(draw_list, lstm_data)
    # draw_heatmap(lstm_data)
    # draw_heatmap(arima_data, 'ARIMA')