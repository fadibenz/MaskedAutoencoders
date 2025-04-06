This is an implementation of "Masked Autoencoders Are Scalable Vision Learners" (MAE) By He et al., I implemented the full paper from scratch using only basic Pytorch building blocks. 


I followed a modular implementation approach, strating with image patchify/unpatchify, the random masking,  transformer part (With learnable positional encoding and CLS token) and finally the full MAE encoder/decoder, you can find each under `ViT/architectures`.

I tested each part rigorously againt the already implemented pytorch versions, you can find the full tests under `notebooks`.

I pretrained the MAE model, and followed two fine-tunning approaches:

1. Linear Probing. 
2. Full finetuning

You can find the training scripts under MAskedAutoencoder/training_scripts and the results with some notes under `notebooks/MAE_final_test`.

This paper allowed me to better grasp the power of pre-training in finding general-purpose non-linear manifolds that can be later on tuned with fine-tunning. Although pretraining is powerful, the poor performance of the first approach showed that the learned representation during pretraining is not general as once thought. 
