1. Running estimate tracking does not work.
2. Decoupling and accumulation of embeddings maybe works.

How to make SSL work (check):
1. Augmentation types
2. Number of views (invariance loss)
3. Batch size (distribution regularization)
4. Projection dimension (just enough so that the rank won't go to max too quickly)
5. Learning rate (too high and it won't learn at all)
6. Lambda (regularization strength)
7. Others like weight decay etc.