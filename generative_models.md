Based on the training strategies, deep molecular generative models can be classified into two categories: reinforcement learning (RL)-based methods, which generate molecules with desired properties; unsupervised (UL)-based or self-supervised (SSL)-based methods, which aim to generate valid, novel, and diverse molecules; supervised (SL)-based methods generating molecular three-dimensional conformations from molecular graphs. 

## RL-Based Generators

- [NeurIPS 2018] [GCPN] You, Jiaxuan, et al. "Graph convolutional policy network for goal-directed molecular graph generation." Advances in neural information processing systems 31 (2018). [[paper]](https://proceedings.neurips.cc/paper/2018/hash/d60678e8f2ba9c540798ebbde31177e8-Abstract.html) [[code]](https://github.com/bowenliu16/rl_graph_generation) 

- [Sci. Adv. 2018] [ReLeaSE] Popova, Mariya, Olexandr Isayev, and Alexander Tropsha. "Deep reinforcement learning for de novo drug design." Science advances 4.7 (2018): eaap7885. [[paper]](https://www.science.org/doi/10.1126/sciadv.aap7885) [[code]](https://github.com/isayev/ReLeaSE)



## SL-Based Generator - Molecular Conformation

- [ICLR 2022 (Oral)] [GeoDiff] Xu, Minkai, et al. "Geodiff: A geometric diffusion model for molecular conformation generation." arXiv preprint arXiv:2203.02923 (2022). [[paper]](https://openreview.net/forum?id=PzcvxEMzvQC) [[code]](https://github.com/MinkaiXu/GeoDiff) 

- [NeurIPS 2022] [torsional diffusion] Jing, Bowen, et al. "Torsional diffusion for molecular conformer generation." arXiv preprint arXiv:2206.01729 (2022). [[paper]](https://arxiv.org/abs/2206.01729) [[code]](https://github.com/gcorso/torsional-diffusion) 

- [TMLR 2022] [DMCG] Zhu, Jinhua, et al. "Direct molecular conformation generation." arXiv preprint arXiv:2202.01356 (2022). [[paper]](https://arxiv.org/abs/2202.01356) [[code]](https://github.com/DirectMolecularConfGen/DMCG) 

- [NeurIPS 2021] [GeoMol] Ganea, Octavian, et al. "Geomol: Torsional geometric generation of molecular 3d conformer ensembles." Advances in Neural Information Processing Systems 34 (2021): 13757-13769. [[paper]](https://proceedings.neurips.cc/paper/2021/hash/725215ed82ab6306919b485b81ff9615-Abstract.html) [[code]](https://github.com/PattanaikL/GeoMol)

- [NeurIPS 2021] [DGSM] Luo, Shitong, et al. "Predicting molecular conformation via dynamic graph score matching." Advances in Neural Information Processing Systems 34 (2021): 19784-19795. [[paper]](https://proceedings.neurips.cc/paper/2021/hash/a45a1d12ee0fb7f1f872ab91da18f899-Abstract.html) 😢 No official codes are available.

- [ICML 2021] [ConfGF] Shi, Chence, et al. "Learning gradient fields for molecular conformation generation." International Conference on Machine Learning. PMLR, 2021. [[paper]](http://proceedings.mlr.press/v139/shi21b.html) [[code]](https://github.com/DeepGraphLearning/ConfGF) 

- [ICML 2021] [ConfVAE] Xu, Minkai, et al. "An end-to-end framework for molecular conformation generation via bilevel programming." International Conference on Machine Learning. PMLR, 2021. [[paper]](http://proceedings.mlr.press/v139/xu21f.html) [[code]](https://github.com/MinkaiXu/ConfVAE-ICML21) 

- [ICLR 2021] [CGCF] Xu, Minkai, et al. "Learning neural generative dynamics for molecular conformation generation." arXiv preprint arXiv:2102.10240 (2021). [[paper]](https://arxiv.org/abs/2102.10240) [[code]](https://github.com/DeepGraphLearning/CGCF-ConfGen) 

- [NeurIPS 2020] [TorsionNet] Gogineni, Tarun, et al. "Torsionnet: A reinforcement learning approach to sequential conformer search." Advances in Neural Information Processing Systems 33 (2020): 20142-20153. [[paper]](https://proceedings.neurips.cc/paper/2020/hash/e904831f48e729f9ad8355a894334700-Abstract.html) [[code]](https://github.com/tarungog/torsionnet_paper_version) 

- [ICML 2020] [GraphDG] Simm, Gregor NC, and José Miguel Hernández-Lobato. "A generative model for molecular distance geometry." arXiv preprint arXiv:1909.11459 (2019). [[paper]](https://arxiv.org/abs/1909.11459) [[code]](https://github.com/gncs/graphdg) 

- [Sci. Rep. 2019] [CVGAE] Mansimov, Elman, et al. "Molecular geometry prediction using a deep generative graph neural network." Scientific reports 9.1 (2019): 20381. [[paper]](https://www.nature.com/articles/s41598-019-56773-5) [[code]](https://github.com/nyu-dl/dl4chem-geometry) 



## UL-Based & SSL-Based Generator - Molecular Graph

- [ICLR 2022 (Oral)] [DEG] Guo, Minghao, et al. "Data-efficient graph grammar learning for molecular generation." arXiv preprint arXiv:2203.08031 (2022). [[paper]](https://openreview.net/forum?id=l4IHywGq6a) [[code]](https://github.com/gmh14/data_efficient_grammar) 

- [ICLR 2022 (Spotlight)] [STGG] Ahn, Sungsoo, et al. "Spanning tree-based graph generation for molecules." International Conference on Learning Representations. 2021. [[paper]](https://openreview.net/forum?id=w60btE_8T2m) 😢 No official codes are available. 

- [ICML 2021] [GraphDF] Luo, Youzhi, Keqiang Yan, and Shuiwang Ji. "Graphdf: A discrete flow model for molecular graph generation." International Conference on Machine Learning. PMLR, 2021. [[paper]](https://proceedings.mlr.press/v139/luo21a.html) [[code]](https://github.com/lakshayguta/BTP/tree/378aac3ae9620aac43a995bcbfb71288593a04c9/DIG-main/dig/ggraph/GraphDF) 

- [ICML 2020] [RationaleRL] Jin, Wengong, Regina Barzilay, and Tommi Jaakkola. "Multi-objective molecule generation using interpretable substructures." International conference on machine learning. PMLR, 2020. [[paper]](https://proceedings.mlr.press/v119/jin20b.html) [[code]](https://github.com/wengong-jin/multiobj-rationale) 

- [ICLR 2020] [GraphAF] Shi, Chence, et al. "Graphaf: a flow-based autoregressive model for molecular graph generation." arXiv preprint arXiv:2001.09382 (2020). [[paper]](https://arxiv.org/abs/2001.09382) [[code]](https://github.com/DeepGraphLearning/GraphAF) 

- [ICML 2020] [HierVAE] Jin, Wengong, Regina Barzilay, and Tommi Jaakkola. "Hierarchical generation of molecular graphs using structural motifs." International conference on machine learning. PMLR, 2020. [[paper]](https://proceedings.mlr.press/v119/jin20a.html) [[code]](https://github.com/wengong-jin/hgraph2graph/)

- [arXiv 2019] [GraphNVP] Madhawa, Kaushalya, et al. "Graphnvp: An invertible flow model for generating molecular graphs." arXiv preprint arXiv:1905.11600 (2019). [[paper]](https://arxiv.org/abs/1905.11600) [[code]](https://github.com/pfnet-research/graph-nvp)

- [NeurIPS 2018] [CGVAE] Liu, Qi, et al. "Constrained graph variational autoencoders for molecule design." Advances in neural information processing systems 31 (2018). [[paper]](https://proceedings.neurips.cc/paper/2018/hash/b8a03c5c15fcfa8dae0b03351eb1742f-Abstract.html) [[code]](https://github.com/drigoni/ConditionalCGVAE) 

- [NeurIPS 2018] Ma, Tengfei, Jie Chen, and Cao Xiao. "Constrained generation of semantically valid graphs via regularizing variational autoencoders." Advances in Neural Information Processing Systems 31 (2018). [[paper]](https://proceedings.neurips.cc/paper/2018/hash/1458e7509aa5f47ecfb92536e7dd1dc7-Abstract.html) 😢 No official codes are available. 

- [ICML 2018] [JT-VAE] Jin, Wengong, Regina Barzilay, and Tommi Jaakkola. "Junction tree variational autoencoder for molecular graph generation." International conference on machine learning. PMLR, 2018. [[paper]](https://proceedings.mlr.press/v80/jin18a.html) [[code]](https://github.com/wengong-jin/icml18-jtnn)

- [ICML 2018] [MolGAN] De Cao, Nicola, and Thomas Kipf. "MolGAN: An implicit generative model for small molecular graphs." arXiv preprint arXiv:1805.11973 (2018). [[paper]](https://arxiv.org/abs/1805.11973) [[code]](https://github.com/nicola-decao/MolGAN)



## UL-Based & SSL-Based Generator - SMILES String

- [Chem. Sci. 2021] [STONED] Nigam, AkshatKumar, et al. "Beyond generative models: superfast traversal, optimization, novelty, exploration and discovery (STONED) algorithm for molecules using SELFIES." Chemical science 12.20 (2021): 7079-7090. [[paper]](https://pubs.rsc.org/en/content/articlehtml/2021/sc/d1sc00231g) [[code]](https://github.com/aspuru-guzik-group/stoned-selfies) 

- [arXiv 2018] [ORGAN] Guimaraes, Gabriel Lima, et al. "Objective-reinforced generative adversarial networks (organ) for sequence generation models." arXiv preprint arXiv:1705.10843 (2017). [[paper]](https://arxiv.org/abs/1705.10843) [[code]](https://github.com/gablg1/ORGAN)

- [J Chem Inf Model 2018] [BIMODAL] Grisoni, Francesca, et al. "Bidirectional molecule generation with recurrent neural networks." Journal of chemical information and modeling 60.3 (2020): 1175-1183. [[paper]](https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00943) [[code]](https://github.com/ETHmodlab/BIMODAL) 

- [ACS Cent. Sci. 2018] [VSeq2Seq] Gómez-Bombarelli, Rafael, et al. "Automatic chemical design using a data-driven continuous representation of molecules." ACS central science 4.2 (2018): 268-276. [[paper]](https://pubs.acs.org/doi/10.1021/acscentsci.7b00572) [[unofficial code]](https://github.com/aksub99/molecular-vae) 😢 No official codes are available.

- [ACS Cent. Sci. 2018] Segler, Marwin HS, et al. "Generating focused molecule libraries for drug discovery with recurrent neural networks." ACS central science 4.1 (2018): 120-131. [[paper]](https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00512) [[unofficial code]](https://github.com/jaechanglim/molecule-generator) 😢 No official codes are available.

- [ICML/PMLR 2017] [GVAE] Kusner, Matt J., Brooks Paige, and José Miguel Hernández-Lobato. "Grammar variational autoencoder." International conference on machine learning. PMLR, 2017. [[paper]](https://arxiv.org/abs/1703.01925) [[code]](https://github.com/mkusner/grammarVAE) 



<!-- ## Format

- [conference/journal name year] <MLA cite> [[paper]]() [[code]]() 

-->