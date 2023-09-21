# An Efficient Point Cloud Place Recognition Approach Based on Transformer in Dynamic Environment 

### Introduction
In this paper, we address 3D point cloud place recognition problem for complex dynamic environments, which is particularly relevant for LiDAR-SLAM tasks, such as loop-closure detection and global localization

### Environment and Dependencies
Code was tested using Python 3.8 with PyTorch (1.10 or above) and MinkowskiEngine 0.5.4 on Ubuntu 18.04 with CUDA 11.

The following Python packages are required:
* tqdm
* tensorboardX
* pyyaml
* PyTorch
* MinkowskiEngine (version 0.5.4)
* pytorch_metric_learning (version 1.1 or above)
* pandas
* wandb
* texttable


### Data preprocess

For scene graph data structure, you can can refer to the 'data_process' dir of generating graphs. You need to prepare three types of raw data: point clouds, semantics label, and ground-truth. We also provide the cleared point cloud bin-files with MOS 
[here](https://drive.google.com/file/d/1bm0mBZDZ2r7-l4ENFtEH9-H1jJDADjWp/view?usp=drive_link) 


```bash
data
    |---00
    |    |---000000.json
    |    |---000001.json
    |    |---...
    |
    |---01
    |    |---000000.json
    |    |---000001.json
    |    |---...
    |
    |---...
    |
    |---00.txt
    |---01.txt
    |---...
```


Before the network training or evaluation, run the below code to generate pickles with positive and negative point clouds for each anchor point cloud. This is the same data preprocess as pointNetVLAD
. You can also download training and evaluation datasets from 
[here](https://drive.google.com/open?id=1rflmyfZ1v9cGGH0RL4qXRrKhg-8A-U9q) 

```generate pickles
cd generating_queries/ 

# Generate training tuples for the Refined Dataset
python generate_training_tuples_refine.py --dataset_root <dataset_root_path>

# Generate evaluation tuples
python generate_test_sets.py --dataset_root <dataset_root_path>
```
`<dataset_root_path>` is a path to dataset root folder, e.g. `/data/pointnetvlad/benchmark_datasets/`.
Before running the code, ensure you have read/write rights to `<dataset_root_path>`, as training and evaluation pickles
are saved there. 
### Training

python main_sg.py

### Evaluation

#### For KITTI dataset
You can download pretrained model and evaluation datasets from 
[here](https://drive.google.com/file/d/1XBIbK1K39dloz8yJrVaDzZAfuVU1byUE/view?usp=drive_link). Please note, You need to select different sequences accordingly. The results are it's PR curve and F1 max score. Besides, you need to set the following parameters:
- model: the eval pretrained model file.
- graph_pairs_dir: set to SK label dir ('../PR_DATA/graphs_sk')
- pair_list_dir: set to eval dir which excludes positive pairs, e.g., '../PR_DATA/eval/3_20' 
- sequences: list of test sequences.
- output_path: path to save test results.

```bash
python eval_batch.py
```

#### For Oxford and In-house dataset
You can download training and evaluation datasets from 
[here](https://drive.google.com/file/d/1DzUvNig36jN_jLvt8dG8CDo8ao7CfZvJ/view?usp=drive_link). You need to modify the file path according to your situation. To evaluate pretrained models run the following commands:

```
cd eval
# To evaluate the model trained on the Refined Dataset
python evaluate.py 
```

### License
Our code is released under the MIT License (see LICENSE file for details).
