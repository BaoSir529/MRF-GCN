# MRF-GCN
Prediction model with multi-point relationship fusion via graph convolutional network: a case study on mining-induced surface subsidence

## Requirements
* matplotlib==3.5.2
* numpy==1.21.5
* pandas==1.4.4
* scikit_learn==1.2.2
* seaborn==0.11.2
* torch==1.13.1+cu116

## Usage

* First, installation the [requirements.txt](./requirements.txt) python environment;
* Prepare your dataset and store it in the format in [./data/input.csv](./data/input.csv);
* The correct adjacency matrix is generated based on the point connections and also saved in [./data/praph.csv](./data/graph.csv);
* Run the code with:
```bash
python ./train.py --data_path ./data/input.csv --graph_path ./data/graph.csv --width 5 --epoch 200 --batch_size 10
```
* Or training with the script file [run.sh](./run.sh)
```bash
bash run.sh
```

## Citation

If you use the code in your paper, please kindly star this repo and cite our paper.

