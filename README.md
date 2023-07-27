# Lessons Learned

- **Gradient Accumulation:** When your hardware doesn't have the memory to handle huge batch sizes, use smaller batch sizes and aggregate them and do one optimizer update across all of them. Add to training code.
- **Label Smoothing Loss, constructing custom loss functions:**
  - Construct losses with `nn.Module` same way as making `torch` layers
  - `LabelSmoothingLoss`:
    - Take your predictions, log the probabilities and then use softmax to normalize them
    - Create a mask to ignore terms from the target that are padding
    - Create a distribution of the smoothing probability in non-target coordinates, 1 - smoothing at target coordinates
    - Multiple the probability distribution by the predictions, sum values, cut values by the mask, and average the values
- Custom schedulers, Lambda Schedulers and using them with an optimizer - Schedulers modify learning rate over time, optimizers are what dictate how to update the weights based on the loss
- Filtering pad tokens when calculating accuracy (and loss)
- **Parallelizing computations in MHA:** Instead of making multiple attention layers, you can combine all linear input layers to the attention layers and parallelize all heads at once.
- PyTorch optimizes `ModuleList`s in a `for` loop
- **Making NLP datasets is really annoying:** Just look at the `dataloader` dir LOL
- I did **not** do the steps for sharing weights, nor for BPE. May eventually do this, but unlikely for time
- Make sure you use min_frequency when building vocabs for dataloaders, min_frequency=1 is too compuationally intensiveo
- Using `add_`, `scatter_`, and `fill_` are in place and are more memory efficient, however can only be used in `torch.no_grad` (or other situations that I don't fully know) as they mess with the gradient tracking in PyTorch's computational graph

Citing this PR and this repo for helping understand certain concepts:

https://github.com/jadore801120/attention-is-all-you-need-pytorch/pull/168#