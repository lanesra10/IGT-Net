# Uncertainty Estimation in Intrapartum Ultrasound: The IGT-Net Network

Official pytorch code for "Uncertainty Estimation in Intrapartum Ultrasound: The IGT-Net Network"

- [✔] Code release
- [❌] Paper release

## Abstract
This paper details the IGT-Net architecture, designed for uncertainty estimation in obstetric ultrasound imaging. 
IGT-Net incorporates Spatial Vector Gradient Attention (SVGA) and Neighborhood Attention Transformer (NAT) into InceptionNext, 
achieving high performance in segmenting and locating the pubic symphysis and fetal head. SVGA enhances computational 
efficiency and global feature extraction, while the NAT-based decoder improves the model's perception of segmentation 
targets, especially in complex or noisy environments. Using Test-Time Augmentation (TTA), IGT-Net evaluates variations 
in model output, allowing uncertainty measurement. The experimental results demonstrate the effectiveness of this 
approach with metrics including JS at 89.11%±0.64%, DicePS at 87.64%±0.85%, DiceFH at 94.77%±0.37%, DiceAll at 
94.15%±0.41%, ASDPS at 1.53±0.12, ASDFH at 2.15±0.15, and ASDAll at 1.98±0.13. These scores are among the highest 
in obstetric ultrasound segmentation. In generalization experiments, IGT-Net's uncertainty estimation strategy showed 
excellent segmentation performance across different datasets, indicating robustness and broad applicability in obstetrics. 
Overall, the IGT-Net and TTA combination provides high confidence in segmentation results, offering a significant advancement 
in obstetric ultrasound imaging.

### IGT-Net:

![framework](imgs/IGT-Net.png)

## Performance Comparison

<img src="imgs/performance.png" title="preformance" style="zoom:8%;" align="left"/>


## Environment

- GPU: NVIDIA GeForce RTX3090 GPU
- Pytorch: 1.10.0 cuda 11.4
- cudatoolkit: 11.3.1



