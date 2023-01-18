# SUGAR 
This repository contains the code for [SUGAR: Efficient Subgraph-level Training via Resource-aware Graph Partitioning](https://arxiv.org/abs/2202.00075).

### Prepare
Necessary package to start with: [torch](https://pytorch.org) + [dgl](https://www.dgl.ai/pages/start.html) + [ogb](https://ogb.stanford.edu/docs/home/) + [metis](https://metis.readthedocs.io/en/latest/)
#### Set up the environment
```bash
conda create -n sugar python=3.8
conda activate sugar

# install torch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# install dgl
pip install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html
# (may need this step if 'import dgl' yields error)
conda install cudatoolkit

# install ogb
pip install ogb

# install metis
pip install metis
```

#### Steps to install METIS

(1) Download METIS [here](http://glaros.dtc.umn.edu/gkhome/views/metis), and unzip the file.

(2) Build METIS
```bash
cd metis-5.1.0
make config shared=1
make install
make
```

(3) If ``libmetis.dylib`` or ``libmetis.so`` is not under ``/usr/local/lib``, find it inside directory ``build``,
and set up the METIS_DLL environment
```bash
export METIS_DLL=/usr/local/lib/libmetis.dylib
# or
export METIS_DLL=/your_directory/libmetis.dylib
```

(4) Check METIS is properly installed
```bash
python
>>> import metis
```


### Training
#### ogbn-arxiv
Run baseline GCN with the full graph
```bash
python train.py --run-full
```
Running this command will automatically download the dataset to ``./dataset/``.



Run SUGAR:
```bash
python train.py --first-time 
```
When running the command for first time, adding '--first-time' will first partition the graph, and then start subgraph-level training. 
The graph partitioning results are saved to ``./partition``.

&nbsp;

#### Other graphs
Coming soon