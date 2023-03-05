# Molecular Representation Learning Bank of Papers & Codes

Updating the newest molecular representation learning methods... 

|              | # Paper | Note                                    |
|--------------|---------|-----------------------------------------|
| Point-Based  | 2       | 3D, No bonds are included in encoding   |
| Graph-Based  | 12      | 2D & 3D, Bonds are included in encoding |
| SMILES-Based | 1       | 1D                                      |

Welcome to update the list together! 😉



## Point-Based Methods

- [PMLR 2021] [PaiNN] Schütt, Kristof, Oliver Unke, and Michael Gastegger. "Equivariant message passing for the prediction of tensorial properties and molecular spectra." International Conference on Machine Learning. PMLR, 2021. [[paper]](https://proceedings.mlr.press/v139/schutt21a.html?ref=https://githubhelp.com) [[code]](https://github.com/atomistic-machine-learning/schnetpack)

- [NeurIPS 2017] [SchNet] Schütt, Kristof, et al. "Schnet: A continuous-filter convolutional neural network for modeling quantum interactions." Advances in neural information processing systems 30 (2017). [[paper]](https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html) [[code]](https://github.com/atomistic-machine-learning/SchNet)



## Graph-Based Methods

Because there are lots of graph-based models, we categorize them into supervised learning methods and self-supervised methods. 

### Supervised Learning

- [ICLR 2022] [MolR] Wang, Hongwei, et al. "Chemical-reaction-aware molecule representation learning." arXiv preprint arXiv:2109.09888 (2021). [[paper]](https://arxiv.org/abs/2109.09888) [[code]](https://github.com/hwwang55/MolR) 

- [ICLR 2022] [SphereNet] Liu, Yi, et al. "Spherical message passing for 3d graph networks." arXiv preprint arXiv:2102.05013 (2021). [[paper]](https://arxiv.org/abs/2102.05013) [[code (implemented in DIG library)]](https://github.com/divelab/DIG) 

- [Nat. Mach. Intell. 2022] [GEM] Fang, Xiaomin, et al. "Geometry-enhanced molecular representation learning for property prediction." Nature Machine Intelligence 4.2 (2022): 127-134. [[paper]](https://www.nature.com/articles/s42256-021-00438-4.) [[code]](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/pretrained_compound/ChemRL/GEM) 

- [NeurIPS 2021] [GemNet] Gasteiger, Johannes, Florian Becker, and Stephan Günnemann. "Gemnet: Universal directional graph neural networks for molecules." Advances in Neural Information Processing Systems 34 (2021): 6790-6802. [[paper]](https://proceedings.neurips.cc/paper/2021/hash/35cf8659cfcb13224cbd47863a34fc58-Abstract.html) [[code]](https://github.com/TUM-DAML/gemnet_pytorch)

- [NeurIPS 2020] [DimeNet++] Klicpera, Johannes, et al. "Fast and uncertainty-aware directional message passing for non-equilibrium molecules." arXiv preprint arXiv:2011.14115 (2020). [[paper]](https://arxiv.org/abs/2011.14115) [[code]](https://github.com/gasteigerjo/dimenet)

- [ICLR 2020] [DimeNet] Gasteiger, Johannes, Janek Groß, and Stephan Günnemann. "Directional message passing for molecular graphs." arXiv preprint arXiv:2003.03123 (2020). [[paper]](https://arxiv.org/abs/2003.03123) [[code]](https://github.com/gasteigerjo/dimenet)

- [PMLR 2017] Gilmer, Justin, et al. "Neural message passing for quantum chemistry." International conference on machine learning. PMLR, 2017. [[paper]](https://proceedings.mlr.press/v70/gilmer17a) [[code]](https://github.com/priba/nmp_qc) 



### Self-Supervised Learning

- [ICLR 2022] [GraphMVP] Liu, Shengchao, et al. "Pre-training molecular graph representation with 3d geometry." arXiv preprint arXiv:2110.07728 (2021). [[paper]](https://arxiv.org/abs/2110.07728) [[code]](https://github.com/chao1224/GraphMVP) 

- [NeurIPS 2021] [MGSSL] Zhang, Zaixi, et al. "Motif-based graph self-supervised learning for molecular property prediction." Advances in Neural Information Processing Systems 34 (2021): 15870-15882. [[paper]](https://arxiv.org/abs/2110.00987) [[code]](https://github.com/zaixizhang/MGSSL) 

- [NeurIPS 2020] [GROVER] Rong, Yu, et al. "Self-supervised graph transformer on large-scale molecular data." Advances in Neural Information Processing Systems 33 (2020): 12559-12571. [[paper]](https://proceedings.neurips.cc/paper/2020/hash/94aef38441efa3380a3bed3faf1f9d5d-Abstract.html) [[code]](https://github.com/tencent-ailab/grover)

- [ICLR 2020] [InfoGraph] Sun, Fan-Yun, et al. "Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization." arXiv preprint arXiv:1908.01000 (2019). [[paper]](https://arxiv.org/abs/1908.01000) [[code]](https://github.com/sunfanyunn/InfoGraph) 



### Other Related Works

- [NeurIPS 2020] You, Yuning, et al. "Graph contrastive learning with augmentations." Advances in neural information processing systems 33 (2020): 5812-5823. [[paper]](https://proceedings.neurips.cc/paper/2020/hash/3fe230348e9a12c13120749e3f9fa4cd-Abstract.html) [[code]](https://github.com/Shen-Lab/GraphCL) 

- [ICLR 2020] Hu, Weihua, et al. "Strategies for pre-training graph neural networks." arXiv preprint arXiv:1905.12265 (2019). [[paper]](https://arxiv.org/abs/1905.12265) [[code]](https://github.com/snap-stanford/pretrain-gnns/) 



## SMILES-Based Methods

- [BCB 2019] [SMILES-BERT] Wang, Sheng, et al. "SMILES-BERT: large scale unsupervised pre-training for molecular property prediction." Proceedings of the 10th ACM international conference on bioinformatics, computational biology and health informatics. 2019. [[paper]](https://dl.acm.org/doi/abs/10.1145/3307339.3342186?casa_token=ROSIBxMX2UkAAAAA:q9M-DLpNJozQWqWEABwskuANeWuj8dPhU9ijopTfmnXJw3l7bjUuKEXI-br4yc4PG5cxVU5MT5Y) [[code]](https://github.com/uta-smile/SMILES-BERT)



<!-- ## Format

- [] <MLA cite> [[paper]]() [[code]]() 

-->