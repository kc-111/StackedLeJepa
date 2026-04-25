# Importance of batch size for LeJEPA

**Hypothesis**: 

Batch size matters for the distributional regularization and sliced wassertain distance may be better than Epps-Pulley.

**Finding**: 
1. For the synthetic case, batch size does matter for distributional regularization and more samples helps (either for estimate or for reducing gradient variance). Sliced wassertain distance favors higher frequency features compared the Epps-Pulley statistic test because of the exponential weighting scheme.
2. Although not state of the art, preliminary investigation (in train directory) show that batch size does not matter much for SSL for the imagenette dataset using LeJEPA. Evaluation is done with both KNN and linear probe and show minimal difference when trained until 800 epochs. The LeJEPA paper trained until 100 epochs, maybe the difference closes after 100. However, this finding is confounded by the fact that probing and such is evaluated based on the embeddings and not the projected embeddings where the invariance loss and regularizer applies. The LeJEPA paper theory suggests that isotropic gaussian is the optimal distribution for downstream purposes, but probing results show that if projected output is used, there may be 1-2% decrease in performance for CIFAR10, CIFAR100, and imagenette, which is not that bad compared to prior work where they observed a 10% decrease. However, guillotine regularization is still needed for higher performance.
3. Performance of LeJEPA, like other SSL methods, are heavily dependent on augmentation methods.
4. MoCo style of storing embeddings will not work for LeJEPA unlike for contrastive based methods.
5. Cosine is not as good as MSE in convergence and is slower. Cosine as objective misses out on magnitude information, adding that back in does not lead to better performance than MSE and may need more tuning.
6. Setting a margin doesn't seem to significantly improve things, so it may not be important.

**Conclusion**:

Small batch size tricks such as no grad accumulation of views and samples for distribution estimation and invariance loss is not entirely necessary to reproduce similar performances. However, it may matter if aiming for the highest test score possible, but that would be optimizing to overfit the test case at that point.

**Notes**:
1. Running estimate tracking does not work.
2. Decoupling and accumulation of embeddings maybe works.

How to make SSL work (check):
1. Architecture (no aggressively downsampling/loss in information). Also, numerical bfloat16, float32?
2. Augmentation types
3. Number of views (invariance loss)
4. Batch size (distribution regularization)
5. Projection dimension (just enough so that the rank won't go to max too quickly)
6. Learning rate (too high and it won't learn at all)
7. Lambda (regularization strength)
8. Others like weight decay etc.
