1. Running estimate tracking does not work.
2. Decoupling and accumulation of embeddings maybe works.

How to make SSL work (check):
1. Architecture (no aggressively downsampling/loss in information).
2. Augmentation types
3. Number of views (invariance loss)
4. Batch size (distribution regularization)
5. Projection dimension (just enough so that the rank won't go to max too quickly)
6. Learning rate (too high and it won't learn at all)
7. Lambda (regularization strength)
8. Others like weight decay etc.
