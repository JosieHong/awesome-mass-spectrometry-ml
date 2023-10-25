The molecular representation learning or properties prediction models are categorized as point-based (or quantum-based) methods, graph-based methods, and sequence-based methods. Because the number of graph-based methods is huge, they are further divided into self-supervised learning and supervised learning manners. It is worth noting that the difference between point-based (or quantum-based) methods and graph-based methods is if bonds (i.e. edges) are included in the encoding. 

## Point-Based (or Quantum-Based) Methods 

- [ICLR 2023] Zhou, Gengmo, et al. "Uni-mol: A universal 3d molecular representation learning framework." (2023). [[paper]](https://chemrxiv.org/engage/chemrxiv/article-details/6402990d37e01856dc1d1581) [[code]](https://github.com/dptech-corp/Uni-Mol)

- [PMLR 2021] [PaiNN] Schütt, Kristof, Oliver Unke, and Michael Gastegger. "Equivariant message passing for the prediction of tensorial properties and molecular spectra." International Conference on Machine Learning. PMLR, 2021. [[paper]](https://proceedings.mlr.press/v139/schutt21a.html?ref=https://githubhelp.com) [[code]](https://github.com/atomistic-machine-learning/schnetpack)

- [NeurIPS 2017] [SchNet] Schütt, Kristof, et al. "Schnet: A continuous-filter convolutional neural network for modeling quantum interactions." Advances in neural information processing systems 30 (2017). [[paper]](https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html) [[code]](https://github.com/atomistic-machine-learning/SchNet)



## Graph-Based Methods

Because there are lots of graph-based models, we categorize them into supervised learning methods and self-supervised methods. 

### Self-Supervised Learning

- [ICLR 2023] [Mole-BERT] Xia, Jun, et al. "Mole-bert: Rethinking pre-training graph neural networks for molecules." The Eleventh International Conference on Learning Representations. 2022. [[paper]](https://openreview.net/forum?id=jevY-DtiZTR) [[code]](https://github.com/junxia97/Mole-BERT/tree/2feff8a33e3634b66b7408e2e2780fc9d960909f)

- [ICLR 2023 (spotlight)] [GNS TAT] Zaidi, Sheheryar, et al. "Pre-training via denoising for molecular property prediction." arXiv preprint arXiv:2206.00133 (2022). [[paper]](https://arxiv.org/abs/2206.00133) [[code]](https://github.com/shehzaidi/pre-training-via-denoising) 

- [ICLR 2023] [GeoSSL-DDM] Liu, Shengchao, Hongyu Guo, and Jian Tang. "Molecular geometry pretraining with se (3)-invariant denoising distance matching." arXiv preprint arXiv:2206.13602 (2022). [[paper]](https://arxiv.org/abs/2206.13602) [[code]](https://github.com/chao1224/GeoSSL) 

- [ICLR 2022] [GraphMVP] Liu, Shengchao, et al. "Pre-training molecular graph representation with 3d geometry." arXiv preprint arXiv:2110.07728 (2021). [[paper]](https://arxiv.org/abs/2110.07728) [[code]](https://github.com/chao1224/GraphMVP) 

- [NeurIPS 2021] [MGSSL] Zhang, Zaixi, et al. "Motif-based graph self-supervised learning for molecular property prediction." Advances in Neural Information Processing Systems 34 (2021): 15870-15882. [[paper]](https://arxiv.org/abs/2110.00987) [[code]](https://github.com/zaixizhang/MGSSL) 

- [NeurIPS 2020] [GROVER] Rong, Yu, et al. "Self-supervised graph transformer on large-scale molecular data." Advances in Neural Information Processing Systems 33 (2020): 12559-12571. [[paper]](https://proceedings.neurips.cc/paper/2020/hash/94aef38441efa3380a3bed3faf1f9d5d-Abstract.html) [[code]](https://github.com/tencent-ailab/grover)

- [ICLR 2020] [InfoGraph] Sun, Fan-Yun, et al. "Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization." arXiv preprint arXiv:1908.01000 (2019). [[paper]](https://arxiv.org/abs/1908.01000) [[code]](https://github.com/sunfanyunn/InfoGraph) 



### Supervised Learning

- [NeurIPS 2022] [ComENet] Wang, Limei, et al. "ComENet: Towards Complete and Efficient Message Passing for 3D Molecular Graphs." arXiv preprint arXiv:2206.08515 (2022). [[paper]](https://openreview.net/forum?id=mCzMqeWSFJ) [[code (implemented in DIG library)]](https://github.com/divelab/DIG/blob/b54e27e5660f0a8ba31dbc7d3f056f872b1f3e8e/dig/threedgraph/method/comenet/ocp/README.md) 

- [ICLR 2022] [GNS+Noisy Nodes] Godwin, Jonathan, et al. "Simple GNN regularisation for 3D molecular property prediction & beyond." arXiv preprint arXiv:2106.07971 (2021). [[paper]](https://arxiv.org/abs/2106.07971) [[codes]](https://github.com/Namkyeong/NoisyNodes_Pytorch)

- [ICLR 2022] [MolR] Wang, Hongwei, et al. "Chemical-reaction-aware molecule representation learning." arXiv preprint arXiv:2109.09888 (2021). [[paper]](https://arxiv.org/abs/2109.09888) [[code]](https://github.com/hwwang55/MolR) 

- [ICLR 2022] [SphereNet] Liu, Yi, et al. "Spherical message passing for 3d graph networks." arXiv preprint arXiv:2102.05013 (2021). [[paper]](https://arxiv.org/abs/2102.05013) [[code (implemented in DIG library)]](https://github.com/divelab/DIG) 

- [Nat. Mach. Intell. 2022] [GEM] Fang, Xiaomin, et al. "Geometry-enhanced molecular representation learning for property prediction." Nature Machine Intelligence 4.2 (2022): 127-134. [[paper]](https://www.nature.com/articles/s42256-021-00438-4.) [[code]](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/pretrained_compound/ChemRL/GEM) 

- [Brief. Bioinformatics 2021] [TrimNet] Li, Pengyong, et al. "TrimNet: learning molecular representation from triplet messages for biomedicine." Briefings in Bioinformatics 22.4 (2021): bbaa266. [[paper]](https://academic.oup.com/bib/article-abstract/22/4/bbaa266/5955940) [[code]](https://github.com/yvquanli/TrimNet)

- [NeurIPS 2021] [GemNet] Gasteiger, Johannes, Florian Becker, and Stephan Günnemann. "Gemnet: Universal directional graph neural networks for molecules." Advances in Neural Information Processing Systems 34 (2021): 6790-6802. [[paper]](https://proceedings.neurips.cc/paper/2021/hash/35cf8659cfcb13224cbd47863a34fc58-Abstract.html) [[code]](https://github.com/TUM-DAML/gemnet_pytorch)

- [NeurIPS 2020] [DimeNet++] Klicpera, Johannes, et al. "Fast and uncertainty-aware directional message passing for non-equilibrium molecules." arXiv preprint arXiv:2011.14115 (2020). [[paper]](https://arxiv.org/abs/2011.14115) [[code]](https://github.com/gasteigerjo/dimenet)

- [ICLR 2020] [DimeNet] Gasteiger, Johannes, Janek Groß, and Stephan Günnemann. "Directional message passing for molecular graphs." arXiv preprint arXiv:2003.03123 (2020). [[paper]](https://arxiv.org/abs/2003.03123) [[code]](https://github.com/gasteigerjo/dimenet)

- [Chem. Mater 2019] [MEGNet] Chen, Chi, et al. "Graph networks as a universal machine learning framework for molecules and crystals." Chemistry of Materials 31.9 (2019): 3564-3572. [[paper]](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.9b01294?casa_token=Qt91hGc97ywAAAAA%3A_uRAvtFkZVg-YHOeSw1mgP5K-pHBPqUpErJFugRveatjcHKJzcsoQACGsBbIxXJ0CFrY2Ug2jnXgcA) [[preprint]](https://arxiv.org/abs/1812.05055) [[code]](https://github.com/materialsvirtuallab/megnet)

- [PMLR 2017] Gilmer, Justin, et al. "Neural message passing for quantum chemistry." International conference on machine learning. PMLR, 2017. [[paper]](https://proceedings.mlr.press/v70/gilmer17a) [[code]](https://github.com/brain-research/mpnn) 

- [NeurIPS 2015] [Neural Graph Fingerprints] Duvenaud, David K., et al. "Convolutional networks on graphs for learning molecular fingerprints." Advances in neural information processing systems 28 (2015). [[paper]](https://proceedings.neurips.cc/paper/2015/hash/f9be311e65d81a9ad8150a60844bb94c-Abstract.html) [[code]](https://github.com/HIPS/neural-fingerprint)



### Other Related Works

- [NeurIPS 2020] You, Yuning, et al. "Graph contrastive learning with augmentations." Advances in neural information processing systems 33 (2020): 5812-5823. [[paper]](https://proceedings.neurips.cc/paper/2020/hash/3fe230348e9a12c13120749e3f9fa4cd-Abstract.html) [[code]](https://github.com/Shen-Lab/GraphCL) 

- [ICLR 2020] Hu, Weihua, et al. "Strategies for pre-training graph neural networks." arXiv preprint arXiv:1905.12265 (2019). [[paper]](https://arxiv.org/abs/1905.12265) [[code]](https://github.com/snap-stanford/pretrain-gnns/) 



## Sequence-Based Methods

- [BCB 2019] [SMILES-BERT] Wang, Sheng, et al. "SMILES-BERT: large scale unsupervised pre-training for molecular property prediction." Proceedings of the 10th ACM international conference on bioinformatics, computational biology and health informatics. 2019. [[paper]](https://dl.acm.org/doi/abs/10.1145/3307339.3342186?casa_token=ROSIBxMX2UkAAAAA:q9M-DLpNJozQWqWEABwskuANeWuj8dPhU9ijopTfmnXJw3l7bjUuKEXI-br4yc4PG5cxVU5MT5Y) [[code]](https://github.com/uta-smile/SMILES-BERT)



<!-- ## Format

- [conference/journal name year] <MLA cite> [[paper]]() [[code]]() 

-->