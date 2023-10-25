# Molecular Related Deep Learning Papers & Codes

Updating the molecular-related deep learning methods... 

Collecting rules: 

```markdown
1. Papers without any implementation codes are excluded from this list. 

2. Format: 
    - [conference/journal name year] [(optional) model name] <MLA cite> [[paper]]() [[code]]() 
```



## Statistics

**Molecular representation learning & properties prediction** list is [[here]](prediction_models.md). 

The molecular representation learning or properties prediction models are categorized as point-based (or quantum-based) methods, graph-based methods, and sequence-based methods. Because the number of graph-based methods is huge, they are further divided into self-supervised learning and supervised learning manners. It is worth noting that the difference between point-based (or quantum-based) methods and graph-based methods is if bonds (i.e. edges) are included in the encoding. 

|                                | # Paper | Note                       |
|--------------------------------|---------|----------------------------|
| Point-Based (or Quantum-Based) | 2       | 3D, No bonds are encoded   |
| Graph-Based                    | 20      | 2D & 3D, Bonds are encoded |
| Sequence-Based                 | 1       | 1D                         |

**Molecular generation** list is [[here]](./generative_models.md). 

Based on the training strategies, deep molecular generative models can be classified into two categories: reinforcement learning (RL)-based methods, which generate molecules with desired properties; unsupervised (UL)-based or self-supervised (SSL)-based methods, which aim to generate valid, novel, and diverse molecules; supervised (SL)-based methods generating molecular three-dimensional conformations from molecular graphs. 

|                                                         | # paper  |
|---------------------------------------------------------|----------|
| RL-Based Generator                                      | 2        |
| SL-Based Generator - Molecular Conformation             | 10       |
| UL-Based & SSL-Based Generator - Molecular Graph        | 11       |
| UL-Based & SSL-Based Generator - SMILES String          | 6        |

**Molecular optimization** list is [[here]](./optimization_models.md). 

Coming soon...