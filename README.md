# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Ahmed Nasr's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/ahmednasr7.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| Scaled Dot-Product Attention | Implement GPT-2 scaled dot-product attention from query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/gpt2/gpt2-attention |
| Apply and Reverse Byte-Level BPE | Implement GPT-2 BPE encoding and decoding by applying learned merges and reconstructing the original token sequence. | https://www.tensortonic.com/research/gpt2/gpt2-bpe-encode-decode |
| Train a Byte-Level BPE Vocabulary | Train a GPT-2 byte-pair encoding vocabulary by repeatedly merging the most frequent adjacent token pair. | https://www.tensortonic.com/research/gpt2/gpt2-bpe-training |
| Causal Scaled Dot-Product Attention | Implement GPT-2 causal self-attention with an upper-triangular mask that blocks access to future tokens. | https://www.tensortonic.com/research/gpt2/gpt2-causal-attention |
| GPT-2 Pre-Norm Decoder Block | Assemble a pre-normalized GPT-2 decoder block with causal attention, feed-forward layers, and residual connections. | https://www.tensortonic.com/research/gpt2/gpt2-decoder-block |
| Token and Learned Position Embeddings | Build GPT-2 input representations by summing learned token embeddings with learned positional embeddings. | https://www.tensortonic.com/research/gpt2/gpt2-embedding |
| GPT-2 Position-Wise Feedforward Network | Implement the GPT-2 position-wise feed-forward network with expansion, GELU activation, and output projection. | https://www.tensortonic.com/research/gpt2/gpt2-ffn |
| GPT-2 Full Forward Pass | Build the GPT-2 forward pass from token and position embeddings through decoder blocks to vocabulary logits. | https://www.tensortonic.com/research/gpt2/gpt2-forward |
| GPT-2 GELU Activation | Implement GPT-2's tanh-based GELU approximation for smooth nonlinear activation of feed-forward features. | https://www.tensortonic.com/research/gpt2/gpt2-gelu |
| Greedy Autoregressive Decoding | Implement GPT-2 greedy decoding by repeatedly selecting the highest-logit next token from the model output. | https://www.tensortonic.com/research/gpt2/gpt2-greedy-decode |
| Residual Projection Initialization Scaling | Compute GPT-2 residual projection initialization scaling based on model width and decoder depth. | https://www.tensortonic.com/research/gpt2/gpt2-init-scaling |
| GPT-2 Layer Normalization | Implement GPT-2 layer normalization with per-token statistics and learned scale and bias parameters. | https://www.tensortonic.com/research/gpt2/gpt2-layernorm |
| GPT-2 Multi-Head Causal Attention | Build GPT-2 multi-head attention with QKV projection, causal attention per head, concatenation, and output projection. | https://www.tensortonic.com/research/gpt2/gpt2-mha |
| Temperature and Top-k Sampling | Implement GPT-2 top-k sampling with temperature scaling, probability normalization, and seeded token selection. | https://www.tensortonic.com/research/gpt2/gpt2-topk-sampling |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |
| VAE Decoder | Build a variational autoencoder decoder that maps sampled latent vectors back to reconstructed input probabilities. | https://www.tensortonic.com/research/vae/vae-decoder |
| ELBO Loss Function | Implement the VAE evidence lower bound from reconstruction loss and KL divergence with configurable weighting. | https://www.tensortonic.com/research/vae/vae-elbo-loss |
| VAE Encoder | Implement a variational autoencoder encoder that maps inputs to separate latent mean and log-variance vectors. | https://www.tensortonic.com/research/vae/vae-encoder |
| KL Divergence Loss | Compute the VAE KL-divergence term between a diagonal Gaussian posterior and the standard normal prior. | https://www.tensortonic.com/research/vae/vae-kl-divergence |
| Reparameterization Trick | Implement the VAE reparameterization trick by sampling latent vectors from the predicted mean and log variance. | https://www.tensortonic.com/research/vae/vae-reparameterization |
| Class Token [CLS] | Prepend a learned classification token to each Vision Transformer patch sequence for image-level prediction. | https://www.tensortonic.com/research/vit/vit-class-token |
| ViT Encoder Block | Build a Vision Transformer encoder block with layer normalization, multi-head attention, MLP, and residual connections. | https://www.tensortonic.com/research/vit/vit-encoder-block |
| Complete Vision Transformer | Assemble a complete Vision Transformer from patch embedding, class and position tokens, encoder blocks, and classifier. | https://www.tensortonic.com/research/vit/vit-full-network |
| Classification Head | Implement the Vision Transformer classification head by normalizing and projecting the final class-token representation. | https://www.tensortonic.com/research/vit/vit-mlp-head |
| Patch Embedding | Implement Vision Transformer patch embeddings by splitting images into fixed patches and linearly projecting each patch. | https://www.tensortonic.com/research/vit/vit-patch-embedding |
| Position Embedding | Add learned positional embeddings to Vision Transformer patch-token sequences while preserving batch dimensions. | https://www.tensortonic.com/research/vit/vit-position-embedding |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with explicit handling for zero-norm inputs. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-cosine-similarity |
| Implement Dot Product | Compute the algebraic dot product and geometric angle relationship for two equal-length NumPy vectors. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-dot-product |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors from the square root of summed squared differences. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-euclidean-distance |
| Linear Combination | Compute a weighted linear combination of equal-length NumPy vectors using one aligned scalar coefficient per vector. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-linear-combination |
| Outer Product | Compute the NumPy outer product of two vectors as a matrix containing every pairwise element multiplication. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-outer-product |
| Vector Norms | Compute L1, L2, and infinity norms for a one-dimensional NumPy vector and return them in a float64 array. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-vector-norms |
| Arange and Linspace | Generate a one-dimensional NumPy sequence using either step-based arange or count-based linspace semantics. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-arange-linspace |
| Basic Indexing | Extract a rectangular NumPy subarray with row and column slice boundaries using standard basic indexing. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-basic-indexing |
| Create Arrays from Lists | Create NumPy arrays from Python lists with the requested dtype and return their values, shape, dimensions, and element count. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-create-array |
| Random Array Generation | Generate seeded float64 NumPy arrays from either a uniform or standard normal distribution. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-random-arrays |
| Reshaping Arrays | Transform a float64 NumPy array with flattening, transposition, or a validated target shape. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-reshape |
| Zeros and Ones | Create a two-dimensional float64 NumPy array of a requested shape filled entirely with zeros or ones. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-zeros-ones |
| Activation Functions | Implement four common activation functions from scratch using basic PyTorch tensor operations (no torch.nn module). | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-activation-function-from-scratch |
| Basic Autograd | Use PyTorch autograd to evaluate a scalar function and return its derivative at every supplied input value. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-basic-autograd |
| Simple Neural Network | Implement a class SimpleNet subclassing nn.Module with two linear layers and ReLU between them. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-build-simple-nn-from-scratch |
| Custom Linear Layer | Implement a custom linear layer that computes the affine transformation without using any built-in linear layer. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-custom-linear-layer |
| Tensor Operations | Perform common element-wise and matrix tensor operations: add, multiply, matmul, power, and max. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-tensor-arithmetic |
| Tensor Factory | Create PyTorch tensors with zeros, ones, or a constant fill value using the requested shape and dtype. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-tensor-creation |
| Tensor Shape Manipulation | Reshape tensors using three common PyTorch operations: flatten to collapse into 1D, squeeze to remove size-1 dimensions. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-tensor-reshape |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/ahmednasr7)
<!-- tensortonic:end -->
