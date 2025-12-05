# llm-from-scratch

A minimal, educational implementation of a GPT-style autoregressive language model built and trained from scratch using PyTorch.

This project does not rely on existing pretrained LLMs.

Instead, it implements the entire pipeline end-to-end:

Custom GPT model (embeddings → multi-head attention → feed-forward → logits)

Tokenization using GPT-2 tokenizer

Sliding-window dataset loader

Training loop with AMP, gradient clipping, LR warmup, checkpoints

Text generation with temperature, top-k, top-p sampling

The model was trained on ~120M tokens of English text (Gutenberg + WikiText subset) and produces coherent short generations.

✨ Features

Pure PyTorch implementation

GPT architecture (256-dim, 4 layers, 4 heads)

Efficient dataloader with stride-based windowing

Automatic mixed precision (AMP)

Checkpointing + resume training

Text generation with repetition penalty, top-k/top-p

Simple training and inference scripts

📈 Evaluation Results :

Validation on data/val.txt (≈9M tokens windowed):

Checkpoint	Avg Loss	  Perplexity

epoch_4	    ~3.99	       ~54

epoch_5	  ~3.98–4.01	   ~54–55





