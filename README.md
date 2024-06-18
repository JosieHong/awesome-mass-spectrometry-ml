# Awesome Machine Learning in Small Molecules Mass Spectrometry

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[<img src="./logo.png" align="right" width="100">](https://github.com/JosieHong/awesome-mass-spectrometry-ml "from emoji kitchen")

> Mass spectrometry, also called mass spec, is an analytical technique that is used to measure the mass-to-charge ratio of ions. The results are presented as a mass spectrum, a plot of intensity as a function of the mass-to-charge ratio.  
> 
> *from [Wikipedia](https://en.wikipedia.org/wiki/Mass_spectrometry)*

Keep updating the awesome machine-learning papers and codes related to small molecules mass spectrometry. Please notice that awesome lists are curations of the best, not everything. [Contributes](contributing.md) are always welcome! 



## Contents

* [Databases](#databases)
* [Papers](#papers)
    * [Survey papers](#survey-papers)
    * [Small molecular representation learning](#small-molecular-representation-learning)
        * [Discussions in pre-train models](#discussion-in-pretrain-models)
        * [Point-based (or quantum-based) methods](#point-based-or-quantum-based-methods)
        * [Graph-based methods](#graph-based-methods)
        * [Sequence-based methods](#sequence-based-methods)
    * [Mass spectrometry-related properties prediction](#ms_prop_prediction)
        * [Tandem mass spectra prediction](#msms-predicton)
        * [Retention time prediction](#retetntion-time-prediction)
        * [Collision cross section prediction](#collision-cross-section-prediction)
    * [Mass spectra representation learning and matching](#mass-spectra-representation-learning-and-matching)
    * [Chemical formula prediction from mass spectra](#chemical-formula-prediction-from-mass-spectra)
* [Machine learning in small molecules chromatography](#machine-learning-in-small-molecules-chromatography)
* [Related awesome lists](#related-awesome-lists)



## Databases

**(Quantum-related) molecular properties**: 

- [OC20 & OC22](https://opencatalystproject.org/): The Open Catalyst Project focuses on using AI to find new renewable energy storage catalysts, releasing the OC20 and OC22 datasets with 1.3 million molecular relaxations from 260 million DFT calculations for research support. 
- [QM9](https://www.nature.com/articles/sdata201422): This dataset includes the computed geometric, energetic, electronic, and thermodynamic properties of 134,000 stable small organic molecules composed of CHONF.
- [GEOM](https://nature.com/articles/s41597-022-01288-4): This dataset features 37 million molecular conformations for over 450,000 molecules, generated using advanced sampling and semi-empirical density functional theory (DFT).
- [MD17 & MD22](http://www.sgdml.org/): The MD22 benchmark dataset includes molecular dynamics trajectories of seven biomolecular and supramolecular systems, with atom counts ranging from 42 to 370, sampled at 400-500 K with 1 fs resolution, and energy and forces calculated using PBE+MBD theory. 
- [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/): PCQM4Mv2 is a quantum chemistry dataset derived from the PubChemQC project, focusing on the ML task of predicting DFT-calculated HOMO-LUMO energy gaps of molecules using their 2D graphs, a significant task due to the expense of obtaining 3D equilibrium structures. 
- [MoleculeNet](https://moleculenet.org/): MoleculeNet is a benchmark for testing machine learning methods on molecular properties, featuring over 700,000 compounds from multiple databases, integrated into the DeepChem package, and evaluates model performances using metrics like AUC-ROC, AUC-PRC, RMSE, and MAE. 

**MS/MS**: 

- [NIST23](https://www.sisweb.com/software/nist-msms.htm): The NIST MS/MS Library 2023 is a collection of MS/MS spectra and search software. It contains 2,374,064 MS/MS spectra from 399,267 small molecules.
- [MoNA](https://mona.fiehnlab.ucdavis.edu/): MoNA currently contains 2,061,612 mass spectral records from experimental and in-silico libraries, as well as from user contributions.
- [GNPS](https://gnps.ucsd.edu/ProteoSAFe/static/gnps-splash.jsp): GNPS is a web-based mass spectrometry ecosystem that aims to be an open-access knowledge base for the community-wide organization and sharing of raw, processed, or annotated fragmentation mass spectrometry data (MS/MS).

**Retention time**: 

- [SMRT](https://www.nature.com/articles/s41467-019-13680-7): This dataset presents an experimentally acquired reverse-phase chromatography retention time dataset, covering up to 80,038 small molecules.
- [RepoRT](https://chemrxiv.org/engage/chemrxiv/article-details/64a5a08c9ea64cc1677e120f): RepoRT currently contains 373 datasets, 8,809 unique compounds, and 88,325 retention time entries measured on 49 different chromatographic columns using various eluents, flow rates, and temperatures.

**Collision cross section**: 

- [AllCCS](https://www.nature.com/articles/s41467-020-18171-8): This collection includes more than 5,000 experimental CCS records and approximately 12 million calculated CCS values for over 1.6 million small molecules.
- [AllCCS2](https://pubs.acs.org/doi/10.1021/acs.analchem.3c02267): Compared to AllCCS, AllCCS2 incorporates newly available experimental CCS data, including 10,384 records from 4,326 compounds. After standardization, 7,713 unified CCS values with confidence scores were added. 
- [METLIN-CCS](https://www.nature.com/articles/s41592-023-02078-5): The METLIN-CCS database includes collision cross section (CCS) values derived from IMS data for more than 27,000 molecular standards across 79 chemical classes. 



## Papers

### Survey papers

- [TrAC 2021] Debus, Bruno, et al. [Deep learning in analytical chemistry](https://www.sciencedirect.com/science/article/pii/S016599362100282X)
- [IJCAI 2023] Xia, Jun, et al. [A Systematic Survey of Chemical Pre-trained Models](https://www.ijcai.org/proceedings/2023/760)
- [J. Cheminform. 2013] Scheubert, Kerstin, et al. [Computational mass spectrometry for small molecules](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-5-12)

### Small molecular representation learning

According to the information embedded in the model, the molecular representation learning models are categorized as point-based (or quantum-based) methods, graph-based methods, and sequence-based methods. Because the number of graph-based methods is huge, they are further divided into self-supervised learning and supervised learning manners. It is worth noting that the difference between point-based (or quantum-based) methods and graph-based methods is if bonds (i.e. edges) are included in the encoding. 

**Discussions in pre-train models** <a id="discussion-in-pretrain-models"></a>

- [JCIM 2023] Zhang, Ziqiao, et al. [Can Pre-trained Models Really Learn Better Molecular Representations for AI-aided Drug Discovery?](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01707)
- [NeurIPS 2022] Sun, Ruoxi, et al. [Does GNN Pretraining Help Molecular Representation?](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html)

**Point-based (or quantum-based) methods** <a id="point-based-or-quantum-based-methods"></a>

- [ICLR 2023] Zhou, Gengmo, et al. [Uni-mol: A universal 3d molecular representation learning framework](https://chemrxiv.org/engage/chemrxiv/article-details/6402990d37e01856dc1d1581) [\[code\]](https://github.com/dptech-corp/Uni-Mol)
- [PMLR 2021] Schütt, Kristof, et al. [Equivariant message passing for the prediction of tensorial properties and molecular spectra](https://proceedings.mlr.press/v139/schutt21a.html?ref=https://githubhelp.com) [\[code\]](https://github.com/atomistic-machine-learning/schnetpack)
- [NeurIPS 2017] Schütt, Kristof, et al. [Schnet: A continuous-filter convolutional neural network for modeling quantum interactions](https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html) [\[code\]](https://github.com/atomistic-machine-learning/SchNet)

**Graph-based methods** <a id="graph-based-methods"></a>

*Self-Supervised Learning:*

- [Brief. Bioinformatics 2024] Zhen, Wang, et al. [BatmanNet: bi-branch masked graph transformer autoencoder for molecular representation](https://academic.oup.com/bib/article/25/1/bbad400/7455246) [\[code\]](https://github.com/wz-create/BatmanNet)
- [Bioinformatics 2023] [3DGCL] Moon, Kisung, et al. [3D graph contrastive learning for molecular property prediction](https://academic.oup.com/bioinformatics/article/39/6/btad371/7192173) [\[code\]](https://github.com/moonkisung/3DGCL)
- [ICLR 2023] [Mole-BERT] Xia, Jun, et al. [Mole-bert: Rethinking pre-training graph neural networks for molecules](https://openreview.net/forum?id=jevY-DtiZTR) [\[code\]](https://github.com/junxia97/Mole-BERT/tree/2feff8a33e3634b66b7408e2e2780fc9d960909f)
- [ICLR 2023 (spotlight)] [GNS TAT] Zaidi, Sheheryar, et al. [Pre-training via denoising for molecular property prediction](https://arxiv.org/abs/2206.00133) [\[code\]](https://github.com/shehzaidi/pre-training-via-denoising)
- [ICLR 2023] [GeoSSL-DDM] Liu, Shengchao, et al. [Molecular geometry pretraining with se (3)-invariant denoising distance matching](https://arxiv.org/abs/2206.13602) [\[code\]](https://github.com/chao1224/GeoSSL)
- [ICLR 2022] [GraphMVP] Liu, Shengchao, et al. [Pre-training molecular graph representation with 3d geometry](https://arxiv.org/abs/2110.07728) [\[code\]](https://github.com/chao1224/GraphMVP)
- [NeurIPS 2021] [MGSSL] Zhang, Zaixi, et al. [Motif-based graph self-supervised learning for molecular property prediction](https://arxiv.org/abs/2110.00987) [\[code\]](https://github.com/zaixizhang/MGSSL)
- [NeurIPS 2020] [GROVER] Rong, Yu, et al. [Self-supervised graph transformer on large-scale molecular data](https://proceedings.neurips.cc/paper/2020/hash/94aef38441efa3380a3bed3faf1f9d5d-Abstract.html) [\[code\]](https://github.com/tencent-ailab/grover)
- [ICLR 2020] [InfoGraph] Sun, Fan-Yun, et al. [Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization](https://arxiv.org/abs/1908.01000) [\[code\]](https://github.com/sunfanyunn/InfoGraph)

*Supervised Learning*

- [AAAI 2023] [Molformer] Wu, Fang, et al. [Molformer: Motif-based transformer on 3d heterogeneous molecular graphs](https://ojs.aaai.org/index.php/AAAI/article/view/25662) [\[code\]](https://github.com/smiles724/Molformer/tree/master)
- [NeurIPS 2022] [ComENet] Wang, Limei, et al. [ComENet: Towards Complete and Efficient Message Passing for 3D Molecular Graphs](https://openreview.net/forum?id=mCzMqeWSFJ) [\[code (implemented in DIG library)\]](https://github.com/divelab/DIG/blob/b54e27e5660f0a8ba31dbc7d3f056f872b1f3e8e/dig/threedgraph/method/comenet/ocp/README.md)
- [ICLR 2022] [GNS+Noisy Nodes] Godwin, Jonathan, et al. [Simple GNN regularisation for 3D molecular property prediction & beyond](https://arxiv.org/abs/2106.07971) [\[codes\]](https://github.com/Namkyeong/NoisyNodes_Pytorch)
- [ICLR 2022] [MolR] Wang, Hongwei, et al. [Chemical-reaction-aware molecule representation learning](https://arxiv.org/abs/2109.09888) [\[code\]](https://github.com/hwwang55/MolR)
- [ICLR 2022] [SphereNet] Liu, Yi, et al. [Spherical message passing for 3d graph networks](https://arxiv.org/abs/2102.05013) [\[code (implemented in DIG library)\]](https://github.com/divelab/DIG)
- [Nat. Mach. Intell. 2022] [GEM] Fang, Xiaomin, et al. [Geometry-enhanced molecular representation learning for property prediction](https://www.nature.com/articles/s42256-021-00438-4) [\[code\]](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/pretrained_compound/ChemRL/GEM)
- [NeurIPS 2021] [GemNet] Gasteiger, Johannes, et al. [Gemnet: Universal directional graph neural networks for molecules](https://proceedings.neurips.cc/paper/2021/hash/35cf8659cfcb13224cbd47863a34fc58-Abstract.html) [\[code\]](https://github.com/TUM-DAML/gemnet_pytorch)
- [NeurIPS 2020] [DimeNet++] Klicpera, Johannes, et al. [Fast and uncertainty-aware directional message passing for non-equilibrium molecules](https://arxiv.org/abs/2011.14115) [\[code\]](https://github.com/gasteigerjo/dimenet)
- [ICLR 2020] [DimeNet] Gasteiger, Johannes, et al. [Directional message passing for molecular graphs](https://arxiv.org/abs/2003.03123) [\[code\]](https://github.com/gasteigerjo/dimenet)
- [Chem. Mater 2019] [MEGNet] Chen, Chi, et al. [Graph networks as a universal machine learning framework for molecules and crystals](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.9b01294) [\[preprint\]](https://arxiv.org/abs/1812.05055) [\[code\]](https://github.com/materialsvirtuallab/megnet)
- [PMLR 2017] Gilmer, Justin, et al. [Neural message passing for quantum chemistry](https://proceedings.mlr.press/v70/gilmer17a) [\[code\]](https://github.com/brain-research/mpnn)
- [NeurIPS 2015] [Neural FPs] Duvenaud, David K., et al. [Convolutional networks on graphs for learning molecular fingerprints](https://proceedings.neurips.cc/paper/2015/hash/f9be311e65d81a9ad8150a60844bb94c-Abstract.html) [\[code\]](https://github.com/HIPS/neural-fingerprint)

*Other Related Works*

- [NeurIPS 2020] You, Yuning, et al. [Graph contrastive learning with augmentations](https://proceedings.neurips.cc/paper/2020/hash/3fe230348e9a12c13120749e3f9fa4cd-Abstract.html) [\[code\]](https://github.com/Shen-Lab/GraphCL)
- [ICLR 2020] Hu, Weihua, et al. [Strategies for pre-training graph neural networks](https://arxiv.org/abs/1905.12265) [\[code\]](https://github.com/snap-stanford/pretrain-gnns/)

**Sequence-based methods** <a id="sequence-based-methods"></a>

- [Patterns 2022] [SELFIES] Krenn, Mario, et al. [SELFIES and the future of molecular string representations](https://www.sciencedirect.com/science/article/pii/S2666389922002069) [\[code\]](https://github.com/aspuru-guzik-group/selfies)
- [Nat. Mach. Intell. 2022] [MolFormer] Ross, Jerret, et al. [Large-scale chemical language representations capture molecular structure and properties](https://www.nature.com/articles/s42256-022-00580-7) [\[code\]](https://github.com/IBM/molformer)
- [BCB 2019] [SMILES-BERT] Wang, Sheng, et al. [SMILES-BERT: large scale unsupervised pre-training for molecular property prediction](https://dl.acm.org/doi/abs/10.1145/3307339.3342186) [\[code\]](https://github.com/uta-smile/SMILES-BERT)



### Mass spectrometry-related properties prediction <a id="ms_prop_prediction"></a>

**Tandem mass spectra prediction predicton** <a id="msms-predicton"></a>

- [Anal. Chem. 2023] Wang, Fei, et al. [Deep Learning-Enabled MS/MS Spectrum Prediction Facilitates Automated Identification Of Novel Psychoactive Substances](https://pubs.acs.org/doi/10.1021/acs.analchem.3c02413) [\[code\]](https://nps-ms.ca/users/sign_in)
- [Nat. Mach. Intell. 2023] Goldman, Samuel, et al. [Annotating metabolite mass spectra with domain-inspired chemical formula transformers](https://www.nature.com/articles/s42256-023-00708-3) [\[code\]](https://github.com/samgoldman97/mist)
- [Nat. Mach. Intell. 2024] Young, Adamo, et al. [Tandem mass spectrum prediction for small molecules using graph transformers](https://arxiv.org/abs/2111.04824) [\[code\]](https://github.com/Roestlab/massformer) 
- [NeurIPS 2023] Goldman, Samuel, et al. [Prefix-tree decoding for predicting mass spectra from molecules](https://arxiv.org/abs/2303.06470) [\[code\]](https://github.com/samgoldman97/ms-pred)
- [Bioinformatics 2023] Hong, Yuhui, et al. [3DMolMS: prediction of tandem mass spectra from 3D molecular conformations](https://academic.oup.com/bioinformatics/article/39/6/btad354/7186501) [\[code\]](https://github.com/JosieHong/3DMolMS)
- [Anal. Chem. 2021] Wang, Fei, et al. [CFM-ID 4.0: more accurate ESI-MS/MS spectral prediction and compound identification](https://pubs.acs.org/doi/full/10.1021/acs.analchem.1c01465) [\[code\]](https://hub.docker.com/r/wishartlab/cfmid)
- [ACS Cent. Sci. 2019] Wei, Jennifer N., et al. [Rapid prediction of electron–ionization mass spectrometry using neural networks](https://pubs.acs.org/doi/full/10.1021/acscentsci.9b00085) [\[code\]](https://github.com/brain-research/deep-molecular-massspec)

**Retetntion time prediction** <a id="retetntion-time-prediction"></a>

- [Bioinformatics 2024] [RT-Transformer] Xue, Jun, et al. [RT-Transformer: Retention time prediction for metabolite annotation to assist in metabolite identification](https://academic.oup.com/bioinformatics/article/40/3/btae084/7613958) [\[code\]](https://github.com/01dadada/RT-Transformer)
- [J. Chromatogr. A 2023] [DeepGCN-RT] Kang, Qiyue, et al. [Deep graph convolutional network for small-molecule retention time prediction](https://www.sciencedirect.com/science/article/pii/S0021967323006647) [\[code\]](https://github.com/kangqiyue/DeepGCN-RT)
- [Anal. Chem. 2021] [GNN-RT] Yang, Qiong, et al. [Prediction of liquid chromatographic retention time with graph neural networks to assist in small molecule identification](https://pubs.acs.org/doi/full/10.1021/acs.analchem.0c04071) [\[code\]](https://github.com/Qiong-Yang/GNN-RT)
- [Anal. Chem. 2020] [Retip] Bonini, Paolo, et al. [Retip: retention time prediction for compound annotation in untargeted metabolomics](https://pubs.acs.org/doi/full/10.1021/acs.analchem.9b05765) [\[code\]](https://www.retip.app/)
- [Nat. Commun 2019] Domingo-Almenara, Xavier, et al. [The METLIN small molecule dataset for machine learning-based retention time prediction](https://www.nature.com/articles/s41467-019-13680-7) [\[code\]](https://figshare.com/articles/dataset/The_METLIN_small_molecule_dataset_for_machine_learning-based_retention_time_prediction/8038913)

**Collision cross section prediction** <a id="collision-cross-section-prediction"></a>

- [Anal. Chem. 2024] de Cripan,  et al. [Predicting the Predicted: A Comparison of Machine Learning-Based Collision Cross-Section Prediction Models for Small Molecules](https://pubs.acs.org/doi/10.1021/acs.analchem.4c00630)
- [Anal. Chem. 2022] [AllCCS2] Zhang, Haosong, et al. [AllCCS2: Curation of Ion Mobility Collision Cross-Section Atlas for Small Molecules Using Comprehensive Molecular Representations](https://pubs.acs.org/doi/full/10.1021/acs.analchem.3c02267) [\[code\]](http://allccs.zhulab.cn/)
- [Anal. Chem. 2022] [CCSP 2.0] Rainey, Markace A., et al. [CCS Predictor 2.0: An open-source jupyter notebook tool for filtering out false positives in metabolomics](https://pubs.acs.org/doi/full/10.1021/acs.analchem.2c03491) [\[code\]](https://github.com/facundof2016/CCSP2.0)
- [Nat. Commun 2020] [AllCCS] Zhou, Zhiwei, et al. [Ion mobility collision cross-section atlas for known and unknown metabolite annotation in untargeted metabolomics](https://www.nature.com/articles/s41467-020-18171-8) [\[code\]](https://github.com/ZhuMetLab/AllCCS)
- [Anal. Chem. 2019] [DeepCCS] Plante, Pier-Luc, et al. [Predicting ion mobility collision cross-sections using a deep neural network: DeepCCS](https://pubs.acs.org/doi/full/10.1021/acs.analchem.8b05821) [\[code\]](https://github.com/plpla/DeepCCS/)

### Mass spectra representation learning and matching

- [Anal. Chem. 2023] [CLERMS] Guo, Hao, et al. [Contrastive learning-based embedder for the representation of tandem mass spectra](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.3c00260) [\[code\]](https://github.com/HaldamirS/CLERMS)
- [Nat. Commun 2023] [FastEI] Yang, Qiong, et al. [Ultra-fast and accurate electron ionization mass spectrum matching for compound identification with million-scale in-silico library](https://www.nature.com/articles/s41467-023-39279-7) [\[code\]](https://github.com/Qiong-Yang/FastEI)
- [PLoS Comput. Biol. 2021] Huber, Florian, et al. [Spec2Vec: Improved mass spectral similarity scoring through learning of structural relationships](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008724) [\[code\]](https://github.com/iomega/spec2vec) 
- [J. Cheminform. 2021] Huber, Florian, et al. [MS2DeepScore: a novel deep learning similarity measure to compare tandem mass spectra](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00558-4) [\[code\]](https://github.com/matchms/ms2deepscore)
- [Anal. Chem. 2019] [DeepMASS] Ji, Hongchao, et al. [Deep MS/MS-aided structural-similarity scoring for unknown metabolite identification](https://pubs.acs.org/doi/10.1021/acs.analchem.8b05405) [\[code\]](https://github.com/hcji/DeepMASS)

### Chemical formula prediction from mass spectra

- [JCIM 2023] Goldman, Samuel, et al. [MIST-CF: Chemical formula inference from tandem mass spectra](https://pubs.acs.org/doi/full/10.1021/acs.jcim.3c01082) [\[code\]](https://github.com/samgoldman97/mist-cf)
- [Nat. Methods 2023] [BUDDY] Xing, Shipei, et al. [BUDDY: molecular formula discovery via bottom-up MS/MS interrogation](https://www.nature.com/articles/s41592-023-01850-x) [\[code1\]](https://github.com/Philipbear/msbuddy) [\[code2\]](https://github.com/Philipbear/BUDDY_Metabolomics) 
- [Nat. Methods 2019] [SIRIUS 4] Dührkop, Kai, et al. [SIRIUS 4: a rapid tool for turning tandem mass spectra into metabolite structure information](https://www.nature.com/articles/s41592-019-0344-8) [\[code\]](https://github.com/boecker-lab/sirius)



## Machine learning in small molecules chromatography

> Mass spectrometry is often coupled with chromatographic techniques, such as GC-MS (gas chromatography-mass spectrometry) or LC-MS (liquid chromatography-mass spectrometry). In these combined techniques, the chromatographic method separates the compounds, and then the mass spectrometer analyzes each separated compound for identification and quantification.

- [Anal. Chem. 2024] [3DMolCSP] Hong, Yuhui, et al. [Enhanced Structure-Based Prediction of Chiral Stationary Phases for Chromatographic Enantioseparation from 3D Molecular Conformations](https://pubs.acs.org/doi/full/10.1021/acs.analchem.3c04028) [\[code\]](https://github.com/JosieHong/3DMolCSP)
- [Nat. Commun 2023] [QGeoGNN] Xu, Hao, et al. [Retention time prediction for chromatographic enantioseparation by quantile geometry-enhanced graph neural network](https://www.nature.com/articles/s41467-023-38853-3) [\[code\]](https://github.com/woshixuhao/Retention-Time-Prediction-for-Chromatographic-Enantioseparation/tree/main/code)
- [J. Sep. Sci. 2018] Piras, Patrick, et al. [Modeling and predicting chiral stationary phase enantioselectivity: An efficient random forest classifier using an optimally balanced training dataset and an aggregation strategy](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/jssc.201701334)
- [J. Chromatogr. A 2016] Sheridan, Robert, et al. [Toward structure-based predictive tools for the selection of chiral stationary phases for the chromatographic separation of enantiomers](https://www.sciencedirect.com/science/article/pii/S0021967316306732) 



## Related awesome lists

- [Awesome Small Molecule Machine Learning](https://github.com/benb111/awesome-small-molecule-ml): This repository focuses on machine learning topics related to small molecules.
- [Awesome Cheminformatics](https://github.com/hsiaoyi0504/awesome-cheminformatics): This repository concentrates on computer-based methods in chemistry.
- [Awesome Python Chemistry](https://github.com/lmmentel/awesome-python-chemistry): This repository is dedicated to Python-based frameworks, libraries, software, and resources in the field of Chemistry.
- [Awesome DeepBio](https://github.com/gokceneraslan/awesome-deepbio) & [deeplearning-biology](https://github.com/hussius/deeplearning-biology): These repositories focus on deep learning methods in biology.
- [awesome-pretrain-on-molecules](https://github.com/junxia97/awesome-pretrain-on-molecules)