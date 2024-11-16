---
layout: post
title: nanoJAXGPT — A pedagogical introduction to JAX/Equinox
date: 2024-10-23 09:00:00
description: Since its introduction, <a href="https://jax.readthedocs.io/en/latest/index.html" target="_blank"><i>JAX</i></a> has seen a significant rise in popularity within the Machine Learning (ML) community. A simple web search would reveal the vast community support, a variety of derivative projects, and a multitude of <i>Python</i> libraries built around <i>JAX</i>. This leads to the inevitable question — What is <i>JAX</i>, and why should I care?
tags: transformers llms python nanoGPT
categories: sample-posts
tabs: false
---

## Introduction

Since its introduction, [_JAX_](https://jax.readthedocs.io/en/latest/index.html) has seen a significant rise in popularity within the Machine Learning (ML) community. A simple web search would reveal the vast community support, a variety of derivative projects, and a multitude of _Python_ libraries built around _JAX_. This leads to the inevitable question: What is _JAX_, and why should I care?

<blockquote>
  Well, according to the official documentation, <i>JAX</i> is a <i>Python</i> library for accelerator-oriented array computation and…</blockquote>

Wait a minute, let’s pump the brakes here! If you were really after the introduction to _JAX_ as outlined in the official docs, you’d be there, not here on this blog post. That being said, while there are plenty of resources to help you kick off your machine learning projects with _JAX_, this article isn’t just about singing praises for _JAX_ as an ML framework nor introducing ML to beginners using it. We’re going to roll up our sleeves and get hands-on, taking a well-known repository (Andrej Karpathy’s _nanoGPT_) and rewriting it from top to bottom using _JAX_ and _Equinox_.

### Umm…_Equinox_?

Yes, if you haven’t heard of this already, _Equinox_ is a library built around _JAX_ with the aim of making the construction of Neural Networks (NN) as smooth as possible. What sets it apart is its familiar _PyTorch_-like syntax, making it a comfortable transition for those coming from a _PyTorch_ background. But don’t be fooled by its simplicity. Underneath the hood, Equinox is diligently registering your model as a [_JAX PyTree_](https://jax.readthedocs.io/en/latest/pytrees.html), a powerful data structure in _JAX_ that allows for complex transformations and computations.

To put it all in context, we’ll illustrate this process through a practical example. Here’s a snippet of code that demonstrates how you can define a Linear layer using _Equinox_:
```python
# Code extracted from https://docs.kidger.site/equinox/all-of-equinox/

import equinox as eqx
import jax

class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias
```

Now, _Equinox_ offers a variety of prebuilt neural network layers, including the _Linear_ layer that we just defined above, that can be utilized to construct complex architectures. A distinctive advantage of _Equinox_ as a library for training deep learning models with _JAX_ is its ability to incorporate arbitrary _Python_ objects, more specifically activation functions, into the _PyTree_ definition. It also provides additional functionality to facilitate the use of _JAX_’s `jax.jit` and `jax.grad` decorators, given that they require all inputs to be _PyTrees_ of arrays, by implementing filtered transformations as `equinox.filter_jit` and `equinox.filter_grad` decorators respectively. You can find more information on filtering in _Equinox_ [here](https://docs.kidger.site/equinox/all-of-equinox/#2-filtering).

## Prerequisites

The following sections of this blog assume that you, the reader possesses a foundational understanding of _JAX_. Below, we compile a comprehensive, yet not exhaustive, list of resources to help you get started.

* JAX introduction tutorial notebooks
  * <a 
    style="
      display: inline-block;
      padding: 6px 12px;
      color: #3a8bdb; 
      border: 1px solid #384148; 
      border-radius: 5px;
      text-decoration: none;
    "
    href="https://jax.readthedocs.io/en/latest/tutorials/">
      Tutorials — JAX documentation
  </a>
* Thinking in JAX
  * <a 
    style="
      display: inline-block;
      padding: 6px 12px;
      color: #3a8bdb; 
      border: 1px solid #384148; 
      border-radius: 5px;
      text-decoration: none;
    "
    href="https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html">
      How to think in JAX — JAX documentation
  </a>
* JAX Automatic vectorization
  * <a 
    style="
      display: inline-block;
      padding: 6px 12px;
      color: #3a8bdb; 
      border: 1px solid #384148; 
      border-radius: 5px;
      text-decoration: none;
    "
    href="https://jax.readthedocs.io/en/latest/automatic-vectorization.html">
      Automatic vectorization — JAX documentation
  </a>
  * <a 
    style="
      display: inline-block;
      padding: 6px 12px;
      color: #3a8bdb; 
      border: 1px solid #384148; 
      border-radius: 5px;
      text-decoration: none;
    "
    href="https://dinocausevic.com/2023/06/13/jax-vmap/">
      JAX VMAP Simplified: An Easy Introduction for Beginners
  </a>
  * <a 
    style="
      display: inline-block;
      padding: 6px 12px;
      color: #3a8bdb; 
      border: 1px solid #384148; 
      border-radius: 5px;
      text-decoration: none;
    "
    href="https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html">
      jax.vmap — JAX documentation
  </a>
* Custom parameter initialization in Equinox
  * <a 
    style="
      display: inline-block;
      padding: 6px 12px;
      color: #3a8bdb; 
      border: 1px solid #384148; 
      border-radius: 5px;
      text-decoration: none;
    "
    href="https://docs.kidger.site/equinox/tricks/#custom-parameter-initialisation">
      Tricks (ensembles, surgery, custom initializations, ...) - Equinox
  </a>


## Notes for Clarity

* In PyTorch, the conventional practice is to define a `forward` method in modules, which is designed to perform actions during the forward pass of the training phase. This approach could be employed in equinox modules as well. However, it is also typical to define the computations for the forward pass within the `__call__` definition of the class. This provides an easy way to define a forward pass for a model, but it’s important to note that any method can be used, and no methods are special-cased. Therefore, in the context of the upcoming sections, when we refer to the forward pass, it is suggested that the reader’s attention be directed towards the `__call__` definition of the respective module, or any other method that the developer chooses to use for this purpose.


## nanoGPT

[nanoGPT](https://github.com/karpathy/nanoGPT) is a simple and fast repository for training or finetuning medium sized GPTs ([Generative Pretrained Transformer](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer)). This is the deep learning repository that we will be rewriting with JAX/Equinox. The contents of this repository is shown in Figure 1 of which we emphasize on `model.py` and `train.py`.


<p align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/647eff9aaa8c04bbf9365219/vEA6dJ6XKpyMWleo5N4KW.png" alt="Description of the image" style="height: 500px;" />
    <br>
    <i>Fig1: Project structure of nanoGPT</i>
</p>

<hr />

### `model.py`

The model outlined in this file draws inspiration from the [GPT-2](https://openai.com/research/gpt-2-1-5b-release) architecture, incorporating various modules to emulate a comparable structure. It is designed to be accessible and comprehensible, even for those new to the field. Let us first outline the most significant modules found in this model definition below.

```python
class CausalSelfAttention(nn.Module):
  ...

class MLP(nn.Module):
  ...

class Block(nn.Module):
  ...

class GPT(nn.Module):
  ...
```

### `train.py`

With the defined model architecture in the `model.py` file, within this file resides a training script to train the model using _PyTorch_. You may observe the contents of this file in the orginal repository linked above. Since the training paradigm in _JAX_ is quite different to that in _PyTorch_, we do not extract and outline the structure of this file here.

## Rewriting `model.py`

### Introducing _SwiGLU_ to _nanoGPT_

In our effort to rewrite _nanoGPT_, we sought to introduce a unique element to the final output. To this end, we implemented the [_SwiGLU_](https://paperswithcode.com/method/swiglu) activation function in place of the standard [_GELU_](https://paperswithcode.com/method/gelu#:~:text=The%20Gaussian%20Error%20Linear%20Unit,standard%20Gaussian%20cumulative%20distribution%20function.) activation within the MLP module. _SwiGLU_, a variant of the [_GLU_](https://paperswithcode.com/method/glu) activation function, is notable for its ability to dynamically adjust non-linearity based on the specific training task. For those interested in delving deeper into _SwiGLU_, additional information can be found [here](https://deci.ai/blog/evolution-of-modern-transformer-swiglu-rope-gqa-attention-is-all-you-need/#:~:text=SwiGLU%3A%20An%20Enhanced%20Activation%20Function%20for%20Better%20Performance).

The mathematical representation of the _SwiGLU_ activation function is as follows:
$$SwiGLU(x, W, V, b, c, \beta) = Swish_{\beta}(xW + b) \otimes (xV + c)$$

Here \\(W, V, b, c\\) are all trainable parameters in the neural network, and we can implement this as shown in the codeblock below. Let us try to breakdown this code step-by-step:

* We first create a subclass of the `eqx.Module` class as this activation function has trainable parameters, and hence we need to register this in our _PyTree_ definition.
* We define the `__init__` method with the three parameters `dim_in`, `dim_out`, and `key`. The first two must be defined during the time of initializing of this module and we will infer the appropriate values based on the input and output number of parameters respectively.
* The `__call__` method implements the definition of the SwiGLU activation function. We apply the [_Swish_](https://paperswithcode.com/method/swish) activation function on one transformation of the input and carry out a component-wise multiplication with another transformation of the input.

```python
import equinox as eqx
import jax
import jax.nn as nn
import jax.numpy as jnp

class SwiGLU(eqx.Module):
    """
    Implementation of the SwiGLU activation function in the paper by Noam Shazeer at Google

    References:
        GLU Variants Improve Transformer paper  : https://arxiv.org/abs/2002.05202
        Aziz et al. Paper Summaries             : https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
    """

    W: jax.Array
    V: jax.Array
    b: jax.Array
    c: jax.Array

    def __init__(self, dim_in, dim_out, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.W = jax.random.normal(k1, (dim_in, dim_out))
        self.V = jax.random.normal(k2, (dim_in, dim_out))
        self.b = jax.random.normal(k3, (dim_out,))
        self.c = jax.random.normal(k4, (dim_out,))

    def __call__(self, x):
        return jax.nn.swish(jnp.dot(x, self.W) + self.b) * (jnp.dot(x, self.V) + self.c)
```

<!-- INFO Block -->
<div 
  style="
    background-color: #1c2b41; 
    border-radius: 5px; 
    padding: 10px;
    margin: 20px 0; 
    display: flex; 
    align-items: flex-start;
  ">
    <img 
      src="https://cdn-uploads.huggingface.co/production/uploads/647eff9aaa8c04bbf9365219/S7GC6ed_inRwpFT2-S4sC.png" 
      alt="Icon" 
      style="
        background-color: transparent;
        margin: 0 10px 0 0;
        height: 25px;
        border: none;
        align-self: flex-start;
      "/>
  <p style="margin: 0; line-height: 1.5; color: #b3bcc9;">
      In most of the upcoming modules, you may notice that there is a <code>config</code> parameter. We pass in a <code>dataclass</code> object initialized from the following <code>GPTConfig</code> definition as an argument to this parameter. It contains a predefined configuration of the architecture of the model.
    </p>
</div>

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
```

### MLP Module

```python
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

Given our gathered experience in constructing a module from scratch with `equinox`, the process of converting the aforementioned _MLP_ layer should be relatively straightforward. We outline the steps for this conversion as follows:

1. Firstly, change this class into an `equinox` module from `torch.nn`.
    ```python
    class MLP(eqx.Module):
    ```
2. Next, let’s rewrite the `__init__` method to initialize the _MLP_ layer in _JAX_. We’ve replaced the _PyTorch_ `nn.Linear` and `nn.Dropout` layers with their _Equinox_ equivalents, keeping the arguments consistent to preserve the original behavior. We initialize the `SwiGLU` module in our _Equinox_ version, carefully selecting the `dim_in` and `dim_out` arguments to match the output dimension of the preceding _Linear_ layer and the input dimension of the subsequent _Linear_ layer, both being `4 * config.n_embd`.
    ```python
    class MLP(eqx.Module):
        c_fc    : eqx.nn.Linear
        swiglu  : SwiGLU
        c_proj  : eqx.nn.Linear
        dropout : eqx.nn.Dropout
    
        def __init__(self, config, key):
            lkey1, lkey2, skey = jax.random.split(key, 3)
    
            self.c_fc     = eqx.nn.Linear(config.n_embd, 4 * config.n_embd, use_bias=config.bias, key=lkey1)
            self.swiglu   = SwiGLU(4 * config.n_embd, 4 * config.n_embd, skey)
            self.c_proj   = eqx.nn.Linear(4 * config.n_embd, config.n_embd, use_bias=config.bias, key=lkey2)
            self.dropout  = eqx.nn.Dropout(config.dropout)
    ```
3. Lastly, we’ve replaced the activation function `self.gelu(x)` with `self.swiglu(x)` in the forward pass. As you may have observed, we have employed a transformation function, `jax.vmap`, during certain steps of the forward pass. This will be further elaborated when we dissect the entire architecture in a layer-by-layer manner, explaining the dimensions of the input that each module receives and the necessity of a `vmap` in such a context.

    However, for the time being, let's continue rewriting the remaining modules in our model.
    ```python
    class MLP(eqx.Module):
        c_fc: eqx.nn.Linear
        swiglu: SwiGLU
        c_proj: eqx.nn.Linear
        dropout: eqx.nn.Dropout
    
        def __init__(self, config, key):
            lkey1, lkey2, skey = jax.random.split(key, 3)
    
            self.c_fc = eqx.nn.Linear(config.n_embd, 4 * config.n_embd, use_bias=config.bias, key=lkey1)
            self.swiglu = SwiGLU(4 * config.n_embd, 4 * config.n_embd, skey)
            self.c_proj = eqx.nn.Linear(4 * config.n_embd, config.n_embd, use_bias=config.bias, key=lkey2)
            self.dropout = eqx.nn.Dropout(config.dropout)
    
        def __call__(self, x):
            x = jax.vmap(self.c_fc)(x)
            x = jax.vmap(self.swiglu)(x)
            x = jax.vmap(self.c_proj)(x)
            x = self.dropout(x)
            return x
    ```

### CausalSelfAttention Module

Moving forward, the process of converting modules should seem fairly straightforward since it mirrors the steps taken in the previous _MLP_ module. We’ll however focus on pointing out the distinct alterations applied in the upcoming module definitions.

#### _PyTorch_ version:
```python
# Code extracted from https://github.com/karpathy/nanoGPT/blob/master/model.py

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
```

#### _Equinox_ version:
```python
class CausalSelfAttention(eqx.Module):
    c_attn: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    attn_dropout: eqx.nn.Dropout
    resid_dropout: eqx.nn.Dropout
    bias: jax.Array = eqx.field(static=True)

    _config: GPTConfig = eqx.field(static=True)

    def __init__(self, config, key):
        assert config.n_embd % config.n_head == 0

        # PRNGKey
        lkey1, lkey2 = jax.random.split(key, 2)

        # key, query, value projections for all heads, but in a batch
        self.c_attn = eqx.nn.Linear(config.n_embd, 3 * config.n_embd, use_bias=config.bias, key=lkey1)
        # output projection
        self.c_proj = eqx.nn.Linear(config.n_embd, config.n_embd, use_bias=config.bias, key=lkey2)
        # regularization
        self.attn_dropout = eqx.nn.Dropout(config.dropout)
        self.resid_dropout = eqx.nn.Dropout(config.dropout)
        self._config = config
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # Has been made a buffer by using lax.stop_gradient whenever it is used.
        # Immutability calls for reshape, plus there is no view for jnp (or numpy) arrays.
        self.bias = jnp.tril(jnp.ones((config.block_size, config.block_size))).reshape(1, 1, config.block_size,
                                                                                       config.block_size)

    def __call__(self, x):
        T, C = jnp.shape(x)  # sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = jnp.split(jax.vmap(self.c_attn)(x), 3, axis=1)
        # Immutability calls for reshape, plus there is no view for jnp (or numpy) arrays.
        k = jnp.swapaxes(k.reshape(T, self._config.n_head, C // self._config.n_head), 0, 1)  # (nh, T, hs)
        q = jnp.swapaxes(q.reshape(T, self._config.n_head, C // self._config.n_head), 0, 1)  # (nh, T, hs)
        v = jnp.swapaxes(v.reshape(T, self._config.n_head, C // self._config.n_head), 0, 1)  # (nh, T, hs)

        # manual implementation of attention
        att = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / math.sqrt(jnp.shape(k)[-1])
        # Note: Added the stop_gradient just to be safe, I see no update rule acting on the bias inside the
        # forward pass.
        att = jnp.where(lax.stop_gradient(self.bias[:, :, :T, :T]) == 0, float('-inf'), att)
        att = jax.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = jnp.matmul(att, v)  # (nh, T, T) x (nh, T, hs) -> (nh, T, hs)
        # Reshaping with Immutability creates a new copy
        y = jnp.swapaxes(y, 1, 2).reshape(T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(jax.vmap(self.c_proj)(y))
        return y
```

* We have rewritten the architecture of this attention module in the `__init__` method to look almost identical, with the exception of the last few lines.
* In this module, along with several subsequent ones, we register the `config` argument as a class field. This is a particular scenario where we are registering a field that does not constitute a layer in the _NN_ architecture. In such a context, it becomes imperative to set it as an _Equinox_ static field using `eqx.field(static=True)`.
* In the forward pass, you’ll notice we’ve changed `B, T, C = x.size()` to `B, T, C = jnp.size(x)`. This is an important difference that highlights the functional programming style of _JAX_. In PyTorch, tensors like `x` are objects with callable methods, so you would call the size method directly on `x`. But in _JAX_, arrays are passed as arguments to functions in `jax.numpy`. As we go through the code, keep an eye out for this functional pattern of passing arrays to _JAX_ functions.

<!-- WARNING Block -->
<div 
  style="
    background-color: #332e1b; 
    border-radius: 5px; 
    padding: 10px;
    margin: 0 0 0 30px; 
    display: flex; 
    align-items: flex-start;
  ">
    <img 
      src="https://img.icons8.com/?size=100&id=5tH5sHqq0t2q&format=png&color=000000" 
      alt="Warning" 
      style="
        background-color: transparent;
        margin: 0 10px 0 0;
        height: 25px;
        border: none;
        align-self: flex-start;
      "/>
    <p style="margin: 0; line-height: 1.5; color: #b3bcc9;">
      It’s important to note that while JAX is rooted in the functional programming paradigm and typically necessitates the passing of JAX arrays into functions as arguments, rather than invoking a method on the array object, it does incorporate certain functionalities as methods of the array for our convenience. A case in point is the <code>jax.numpy.transpose</code> function, which, in addition to its traditional use in functional programming, can also be invoked as a method on the JAX array.
    </p>
</div>


* So here's the deal with `numpy` arrays (and by extension, `jax.numpy` arrays): they don't come with a `view` method attached to them. To get our arrays into the shape we need for the transformations coming up next, we decided to use the handy `jnp.reshape` function.
* In our implementation, we skip the flash attention part and jump right into manually implementing the attention mechanism. You might notice some similarities between our approach and the original, aside from the fact that we're using _JAX's_ functional API.
  * One key difference is that we use the `jnp.matmul` function to perform matrix multiplication, replacing the `@` operator.
  * Another thing to watch out for is that `jnp.transpose` works a bit differently than `torch.transpose`. In _JAX_, `jnp.swapaxes` is the function you'll want to use to achieve the same result as _PyTorch_.


### Block Module

Let’s take a closer look at the Block module, which is a key component of the transformer architecture. You’ll see that it uses almost all of the modules we defined earlier. One thing to note is that in the original _PyTorch_ version, the author of _nanoGPT_ passed in an argument for the `bias` parameter in the _LayerNorm_ layer. If you were a _PyTorch_ veteran (or simply referred the documentation), you might be gather that the built-in _LayerNorm_ module doesn’t actually have this parameter! The author implemented their own custom LayerNorm from scratch to support this optional bias functionality. However, in our rewrite using the _Equinox_ library, the built-in _LayerNorm_ module conveniently includes a `bias` parameter by default, so we can use it directly without needing a custom implementation.

#### _PyTorch_ version:

```python
# Code extracted from https://github.com/karpathy/nanoGPT/blob/master/model.py

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

#### _Equinox_ version:

```python
class Block(eqx.Module):
    ln_1: eqx.nn.LayerNorm
    attn: CausalSelfAttention
    ln_2: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(self, config, key):
        ckey, mkey = jax.random.split(key, 2)

        self.ln_1 = eqx.nn.LayerNorm(config.n_embd, use_bias=config.bias)
        self.attn = CausalSelfAttention(config, ckey)
        self.ln_2 = eqx.nn.LayerNorm(config.n_embd, use_bias=config.bias)
        self.mlp = MLP(config, mkey)

    def __call__(self, x):
        x = x + self.attn(jax.vmap(self.ln_1)(x))
        x = x + self.mlp(jax.vmap(self.ln_2)(x))
        return x
```

### GPT Module

We’ve now reached the top of our model structure. The original version had a lot of methods for this module, more than just the constructor (`__init__`) and `__call__` methods. But, we've cut down most of these methods to keep things simple and focus on the _JAX_ and _Equinox_ parts that we decided to implement in our code.

#### _PyTorch_ version:

```python
# Code extracted from https://github.com/karpathy/nanoGPT/blob/master/model.py

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

       return idx
```

#### _Equinox_ version:

The original codebase defined the transformer layer as a dictionary of Modules (`ModuleDict` from _PyTorch_). However, since in _Equinox_, it is essential that we define the layers of the class as fields just before the constructor, it wasn’t possible to organize the code similar to the original structure. For this reason, as well as simplicity, we extracted the transformer layer into its own module, and we called it `TransformerLayer`.

##### `TransformerLayer` Module

```python
class TransformerLayer(eqx.Module):
    _config: GPTConfig = eqx.field(static=True)

    wte: eqx.nn.Embedding
    wpe: eqx.nn.Embedding
    drop: eqx.nn.Dropout
    h: list
    ln_f: eqx.nn.LayerNorm

    def __init__(self, config, key):
        ekey, pkey, hkey, fkey = jax.random.split(key, 4)

        assert config.vocab_size is not None
        assert config.block_size is not None
        self._config = config

        self.wte = eqx.nn.Embedding(config.vocab_size, config.n_embd, key=ekey)
        self.wpe = eqx.nn.Embedding(config.block_size, config.n_embd, key=pkey)
        self.drop = eqx.nn.Dropout(config.dropout)
        self.h = [Block(config, hkey) for _ in range(config.n_layer)]
        self.ln_f = eqx.nn.LayerNorm(config.n_embd, use_bias=config.bias)

    def __call__(self, idx):
        t, = idx.shape
        assert t <= self._config.block_size, f"Cannot forward sequence of length {t}, block size is only {self._config.block_size}"
        pos = jnp.arange(0, t, dtype=jnp.int64)

        tok_emb = jax.vmap(self.wte)(idx)  # token embeddings of shape (t, n_embd)
        pos_emb = jax.vmap(self.wpe)(pos)  # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = jax.vmap(self.ln_f)(x)

        return x
```

We would like to draw the reader’s attention to the fact that in the first line of the forward pass, we are only capable of unpacking the token dimension length from the input. This is in contrast to the _PyTorch_ implementation where the batch dimension is also obtained. The difference here arises from the fact that we won't be processing a batch of inputs, but instead, a single input containing a sequence of tokens. __DO NOT WORRY!!!__ This will become clear as we construct the training loop, where a vectorized map is applied on the batch dimension.

With the transformer layer in a separate module, the `GPT` module is as simple as it can get. We show you the most minimal version of the `GPT` module below.

##### `GPT` Module

```python
class GPT(eqx.Module):
    _config: GPTConfig = eqx.field(static=True)

    transformer: TransformerLayer
    lm_head: eqx.nn.Linear

    def __init__(self, config, key):
        tkey, lmhkey = jax.random.split(key, 2)

        assert config.vocab_size is not None
        assert config.block_size is not None
        self._config = config

        self.transformer = TransformerLayer(config, tkey)

        self.lm_head = eqx.nn.Linear(config.n_embd, config.vocab_size, use_bias=False, key=lmhkey)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array)))
        if non_embedding:
            n_params -= sum(self.transformer.wpe.weight.shape)
        return n_params
    
    ## CODE STRIPPED FOR DEMONSTRATION
    
    def __call__(self, idx, train_mode=False):
        x = self.transformer(idx)

        if train_mode:
            logits = jax.vmap(self.lm_head)(x)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = jax.vmap(self.lm_head)(x[[-1], :])  # note: using list [-1] to preserve the time dim

        return logits
```

In our `GPT` module's forward pass, you may observe that we don't design the method to take an optional `target` parameter, unlike the _PyTorch_ implementation. In our version, we compute the loss within the training loop. More on that later. However, in this case, we instead accept a parameter to determine the mode in which the forward pass is invoked: training mode or inference mode. As a result, we can implement the appropriate logic during inference time, as seen in the original repo.

Now, it’s only fair we show the reader how we implemented the rest of the logic in the original `GPT` module. We handle this task case-by-case, dividing sections for each method. For each of the methods, we follow a bottom-to-top approach here as well, by showing implementations of the all the dependencies and working our way up.

We first define a helper package in our project to add some of the functional components that will help us implement certain logic in the `GPT` module faster, and more importantly: abstract the logic to bring it closer to _PyTorch_. We define two separate modules within the helper module as follows:

```
.
└── helpers/
    ├── eqx.py
    └── init.py
```

##### `init.py`

```python
def normal_(array: jax.Array, mean: float, std: float, key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> None:
    new_array = jax.random.normal(key, array.shape) * std + mean
    return new_array


def zeros_(array: jax.Array) -> None:
    new_array = jax.numpy.zeros(array.shape)
    return new_array
```

While the second method stands to explain itself on its own, we explain the intent of the first function. It serves the purpose of initializing an input _JAX_ array with a normal distribution with a given standard deviation and mean. This will come to be of use when initializing the `GPT` module.

##### `eqx.py`

```python
def named_parameters(model: eqx.Module):
    out = []

    for path, p in jax.tree_util.tree_flatten_with_path(eqx.filter(model, eqx.is_array))[0]:
        pn = ''

        for index in range(len(path)):
            if isinstance(path[index], str):  # Check if path[index] is a string
                pn += '.' + path[index]
            else:
                pn += str(path[index])

        out.append((pn[1:], p))
    
    return out


def find_sub_tree(model: eqx.Module, sub_tree_name: str, filter_fn: Callable = None):
    out = []
    for path, p in jax.tree_util.tree_flatten_with_path(model, is_leaf=filter_fn)[0]:
        pn = ''
    
        for index in range(len(path)):
            if isinstance(path[index], jax._src.tree_util.DictKey):
                pn += '.' + path[index].key
            else:
                pn += str(path[index])
    
        if filter_fn:
            if filter_fn(p) and pn.endswith(sub_tree_name):
                out.append(p)
        elif pn.endswith(sub_tree_name):
            out.append(p)
    
    return out
```

In this module, the first function is written to replicate the function by the same name available as a method in the class `torch.Module` (read [here](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_parameters)). It takes any _Equinox_ module as an argument and returns a list of tuples, each containing a string representing the path to a parameter in the model and the parameter itself.

The second function can be used to find a parameter whose full name ends with a given string. We shall see how these functions come in handy in just a few more sections.

Circling back to the `GPT` module, focusing on the `_init_weights` method, you may notice that in the _PyTorch_ version, this method serves as a custom initializer for the weights of the _Linear_ and _Embedding_ layers. If you look closely at the constructor, you’ll also see that right after this method is applied to the model, there’s another piece of custom initializer logic. This one is specifically for the residual projection weights (`c_proj.weight`). In our implementation, we’ve combined all these initializer logics into a single function as follows.

##### `_init_weights` `GPT` method

```python
@staticmethod
def _init_weights(model: eqx.Module, config: GPTConfig, key: jax.random.PRNGKey):
    def init_layer(model, is_layer: Callable, mean: float, std: float):
        get_weights = lambda m: [x.weight
                                  for x in jax.tree_util.tree_leaves(m, is_leaf=is_layer)
                                  if is_layer(x)]
        weights = get_weights(model)

        new_weights = [init.normal_(weight, mean=mean, std=std, key=subkey)
                        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]

        return eqx.tree_at(get_weights, model, new_weights)

    def init_linear(model):
        is_linear = lambda x: isinstance(x, eqx.nn.Linear)

        model = init_layer(model, is_linear, mean=0.0, std=0.2)

        get_biases = lambda m: [x.bias
                                for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                                if is_linear(x) and x.bias is not None]
        biases = get_biases(model)

        new_biases = [init.zeros_(bias) for bias in biases]

        return eqx.tree_at(get_biases, model, new_biases)

    def init_embedding(model):
        is_embedding = lambda x: isinstance(x, eqx.nn.Embedding)

        return init_layer(model, is_embedding, mean=0.0, std=0.2)

    def init_c_proj_weights_with_normal(model):
        get_c_proj_weights = lambda m: eqx_helper.find_sub_tree(m, "c_proj.weight")

        old_weights = get_c_proj_weights(model)
        new_weights = [init.normal_(weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer), key=subkey)
                        for weight, subkey in zip(old_weights, jax.random.split(key, len(old_weights)))]

        return eqx.tree_at(get_c_proj_weights, model, new_weights)

    initialized_model = init_linear(model)
    initialized_model = init_embedding(initialized_model)
    # apply special scaled init to the residual projections, per GPT-2 paper
    initialized_model = init_c_proj_weights_with_normal(initialized_model)

    return initialized_model
```

I know! You might be wondering how a few lines of _PyTorch_ code turns into this. I assure you, this will sound simple once we breakdown the code into smaller blocks for explanation. We, however, remind the reader about the immutability of _JAX_ arrays before proceeding. Hence any update to the model cannot be done therein, but instead returned as a new _PyTree_.

__`init_layer`__ This function is written as an abstraction to allow initializing any layer that is filtered through the `is_layer` callable. It will initialize the layers of the input model matching the filter with values sampled from a normal distribution defined by the specified mean and standard deviation. 

<!-- INFO Block -->
<div 
  style="
    background-color: #1c2b41; 
    border-radius: 5px; 
    padding: 10px;
    margin: 20px 0; 
    display: flex; 
    align-items: flex-start;
  ">
    <img 
      src="https://cdn-uploads.huggingface.co/production/uploads/647eff9aaa8c04bbf9365219/S7GC6ed_inRwpFT2-S4sC.png" 
      alt="Icon" 
      style="
        background-color: transparent;
        margin: 0 10px 0 0;
        height: 25px;
        border: none;
        align-self: flex-start;
      "/>
  <p style="margin: 0; line-height: 1.5; color: #b3bcc9;">
      This code is nothing but a simple level of abstraction for the code found in the <i>Equinox</i> documentation for <i>Custom Parameter Initialization</i> (read <a href="https://docs.kidger.site/equinox/tricks/#custom-parameter-initialisation">here</a>). The reader is encouraged to refer to this documentation that we have also listed in our prerequisites section.
    </p>
</div>

__`init_linear`__ Here, we simply call the `init_layer` with the filter to identify _Linear_ layers in the model, and the returned model is then additionally initialized with zeros for the biases of the _Linear_ layers.

__`init_embedding`__ Very similar to the `init_linear` function.

__`init_c_proj_weights_with_normal`__ Achieves the functionality as its name suggests. `c_proj.weights` are initialized with the custom normal distribution.

We call these defined functions and return the new updated model. However, you may have noticed that even though we have defined this `_init_weights` method within the `GPT` module, it is not called in the constructor and hence will not do the necessary update to the model when an instance is created in the traditional sense. To achieve this, we create an additional static method that will be used to create a `GPT` instance with these updated weights.

```python
@staticmethod
def create_instance(config, key):
    key1, key2 = jax.random.split(key, 2)

    inst = GPT(config, key1)
    new_inst = GPT._init_weights(inst, config, key2)

    return new_inst
```

<!-- WARNING Block -->
<div 
  style="
    background-color: #332e1b; 
    border-radius: 5px; 
    padding: 10px;
    margin: 20px 0; 
    display: flex; 
    align-items: flex-start;
  ">
    <img 
      src="https://img.icons8.com/?size=100&id=5tH5sHqq0t2q&format=png&color=000000" 
      alt="Warning" 
      style="
        background-color: transparent;
        margin: 0 10px 0 0;
        height: 25px;
        border: none;
        align-self: flex-start;
      "/>
    <p style="margin: 0; line-height: 1.5; color: #b3bcc9;">
      We avoid using the <code>_init_weight</code> to create the updated instance and simply replace the self object. Instead, we return a new instance that contains the updated weights.
    </p>
</div>

To create a new instance of `GPT`, all we have to do is call `GPT.create_instance` instead of simply `GPT`. With this final method implemented, we come to an end of the `model.py` file. Now, moving onto the `train.py` file, where we show how this model is used in pretraining a language model from scratch.

But first, let us try to understand how the vectorized map works in _JAX_ in the following section. This concept is crucial for a reader to grasp how the training loop is built in the upcoming sections.

<hr />

### Understanding the Vectorized Map (`vmap`) flow

In this section of this blog post, we intend to breakdown the flow of the input data to understand how the `vmap` works in each of the modules from top to bottom. We will use a loosely referenced mathematical notation to make things simpler.

The input into the model will be a batch (ℬ) of tokens (𝒯) representing the text that will be used to pretrain the model.

<!-- INFO Block -->
<div 
  style="
    background-color: #1c2b41; 
    border-radius: 5px; 
    padding: 10px;
    margin: 20px 0; 
    display: flex; 
    align-items: flex-start;
  ">
    <img 
      src="https://cdn-uploads.huggingface.co/production/uploads/647eff9aaa8c04bbf9365219/S7GC6ed_inRwpFT2-S4sC.png" 
      alt="Icon" 
      style="
        background-color: transparent;
        margin: 0 10px 0 0;
        height: 25px;
        border: none;
        align-self: flex-start;
      "/>
  <p style="margin: 0; line-height: 1.5; color: #b3bcc9;">
      This pretraining data can be a dataset of your choice, and you may follow the `prepare.py` scripts within the `data` folder to structure them to our training paradigm.
    </p>
</div>

Hence the input would be a `jnp` array of shape,

  ℬ × 𝒯

Since we will be passing this input to the model in the training script, we will call using the `vmap` transformation on the 0<sup>th</sup> dimension.

<div style="text-align: center;">
  <code>jax.vmap(model, in_axes=(0, None))(x, True)</code>
</div>

In the above code segment, recall that we have to define the batch dimension for every argument we pass into the vmap'd function. Hence, for the argument `x`, we indicate the 0<sup>th</sup> dimension and `None` for the second argument, `True`, to be the batch dimensions respectively.

Now, looking at a very high level, the `GPT` module’s forward method only receives a token stream (𝒯), and the batch is executed parallelly as a series of individual functions.

We then pass this 𝒯 through the transformer as `self.transformer(idx)`.

The first two _Embedding_ layers in the transformer will take in a scalar value and transform it into an embedding vector of the given size. However, we are trying to embed a stream of tokens, 𝒯, to obtain an embedded list of tokens corresponding to our initial input. Therefore, we need to batch `idx` across the 0th dimension so that the _Embedding_ layer will be called with individual scalar values in 𝒯. The resulting array will then be of size 𝒯 × ℰ, _where ℰ is the number of embedding dimensions_.

The same goes for the positional embedding as well. And the resulting array is passed through the `Block` module.

In the `Block`'s forward pass, the layer normalization needs to be carried out on the embedding vector of each token. That is, the token dimension acts as a batch in this case. We apply `vmap` on the 0<sup>th</sup> dimension. The returned array is same as the input.

The reader should now be equipped with sufficient experience to dissect the `vmap` process. We, therefore, leave it for the reader to explore the rest of the `vmap`s as an exercise.

<hr />

## Rewriting `train.py`

Now that we have completed building the model, we can move towards writing the training script. We will focus on the major code segments that lead up to the training process, allowing the rest to be self explanatory.

### `get_batch`

This function will use the prepared bin files for the train/validation sets from executing the relevant dataset script found in the data folder. In our experiments, we execute the `prepare.py` file for the _tinystories_ dataset.

In the following function, we are randomly retrieving a batch of data of the specified size in a format suitable for pretraining the LLM.

<!-- INFO Block -->
<div 
  style="
    background-color: #1c2b41; 
    border-radius: 5px; 
    padding: 10px;
    margin: 20px 0; 
    display: flex; 
    align-items: flex-start;
  ">
    <img 
      src="https://cdn-uploads.huggingface.co/production/uploads/647eff9aaa8c04bbf9365219/S7GC6ed_inRwpFT2-S4sC.png" 
      alt="Icon" 
      style="
        background-color: transparent;
        margin: 0 10px 0 0;
        height: 25px;
        border: none;
        align-self: flex-start;
      "/>
  <p style="margin: 0; line-height: 1.5; color: #b3bcc9;">
      Note that in this training exercise, the original repo intended to use a 600,000 batches to train the model, in contrast to the common convention of epochs.
    </p>
</div>

```python
def get_batch(split: str):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'validation.bin'), dtype=np.uint16, mode='r')

    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = jnp.stack([jnp.array(data[i:i + block_size], dtype=jnp.int64) for i in ix])
    y = jnp.stack([jnp.array(data[i + 1:i + 1 + block_size], dtype=jnp.int64) for i in ix])

    return x, y
```

### `convert_model_to_dtype`

This function serves to convert our model, a _PyTree_, to a specified datatype. Note that we are using the globally defined datatype and simply overriding the global model as well. We call this function after initializing model in any of the three starting states: scratch, resume, or from gpt-2.


```python
def convert_model_to_dtype():
    global model
    def convert_pytree_to_dtype(pytree, dtype):
        def _convert(leaf):
            if eqx.is_array(leaf):
                return leaf.astype(dtype)
            else:
                return leaf
    
        return jax.tree_util.tree_map(_convert, pytree)
    
    
    if dtype == 'bfloat16':
        model = convert_pytree_to_dtype(model, jnp.bfloat16)
    elif dtype == 'float16':
        model = convert_pytree_to_dtype(model, jnp.float16)
    elif dtype == 'float32':
        model = convert_pytree_to_dtype(model, jnp.float32)
```

### `lr_scheduler`

We define a simple cosine decay scheduler for the learning rate as follows. The `decay_steps` is defined so that when the training script is started with the intention of resuming the training process, the scheduler is aware of the remaining number of steps to decay the learning rate across.

<!-- WARNING Block -->
<div 
  style="
    background-color: #332e1b; 
    border-radius: 5px; 
    padding: 10px;
    margin: 20px 0; 
    display: flex; 
    align-items: flex-start;
  ">
    <img 
      src="https://img.icons8.com/?size=100&id=5tH5sHqq0t2q&format=png&color=000000" 
      alt="Warning" 
      style="
        background-color: transparent;
        margin: 0 10px 0 0;
        height: 25px;
        border: none;
        align-self: flex-start;
      "/>
    <p style="margin: 0; line-height: 1.5; color: #b3bcc9;">
      This way of resuming a scheduler is not the most ideal or standard in deep learning practice. However, we proceed with such a rudimentary logic due to an unresolved error we faced while saving the optimizer state, ergo, the learning rate scheduler. We will be most thankful to a curious reader with a solution to saving and resuming the optimizer state of a an `Equinox` model.
    </p>
</div>

```python
lr_scheduler = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=learning_rate,
    warmup_steps=warmup_iters if init_from == 'scratch' else 0,
    decay_steps=lr_decay_iters - iter_num,
    end_value=min_lr,
)
```

### `optimizer`

We define a simple _AdamW_ optimizer with `optax`. We have also used an `optax` wrapper, `inject_hyperparms`, so that we are able to access the current learning rate updated according the scheduler.

```python
optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=lr_scheduler, b1=beta1, b2=beta2)
```

### `compute_loss`

If you recall, we mentioned while defining the forward pass of the `GPT` module, that we will be calculating the loss within the training loop. This loss calculation is defined as a function as shown. This function is _JIT_’d with the `eqx.filter_jit` transformation as we are passing in an _Equinox_ model into it.

```python
@eqx.filter_jit
def compute_loss(model, x, y):
    logits = jax.vmap(model, in_axes=(0, None))(x, True)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, # B, T, C
        labels=y, # B, T
    )

    return jnp.mean(loss)
```

### `make_step`

This is the top level function that is called within the training loop each iteration. This function executes a bunch of crucial steps for the model to train. We will attempt to break it down line-by-line.

```python
@eqx.filter_jit
def make_step(
        model,
        optimizer_state,
        x,
        y
):
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, x, y)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, model)
    model = eqx.apply_updates(model, updates)
    return model, optimizer_state, loss
```

#### Line 1

The `compute_loss` function we wrote earlier is first transformed with the `filter_value_and_grad` function which will calculate the loss as well as the gradients for us. Here, we are conveniently executing the forward and backward pass in one single line.

<!-- INFO Block -->
<div 
  style="
    background-color: #1c2b41; 
    border-radius: 5px; 
    padding: 10px;
    margin: 20px 0; 
    display: flex; 
    align-items: flex-start;
  ">
    <img 
      src="https://cdn-uploads.huggingface.co/production/uploads/647eff9aaa8c04bbf9365219/S7GC6ed_inRwpFT2-S4sC.png" 
      alt="Icon" 
      style="
        background-color: transparent;
        margin: 0 10px 0 0;
        height: 25px;
        border: none;
        align-self: flex-start;
      "/>
  <p style="margin: 0; line-height: 1.5; color: #b3bcc9;">
      The <code>eqx.filter_value_and_grad</code> function is <i>Equinox</i>’s implementation of the <code>jax.value_and_grad</code> transformation to account for the non <i>JAX</i> arrays present in the model. 
    </p>
</div>

#### Line 2

With the calculated gradients, we compute the necessary updates for the model with the current optimizer state.

#### Line 3

The calculated updates are now applied to the model. This is the actual step that is taken towards reducing the model loss affected from the parameters.

#### Line 4

The updated model, optimizer state and the loss before making the step is returned to be accessed from the training loop.

### `estimate_loss`

This function is written to calculate the training and evaluation loss at a fixed interval determined according to the training setup and is executed within the train loop.

```python
def estimate_loss(model):
    out = {}
    model = eqx.nn.inference_mode(model)
    for split in ['train', 'val']:
        losses = jnp.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            loss = compute_loss(model, jax.lax.stop_gradient(x), y)
            losses = losses.at[k].set(loss.item())
        out[split] = jnp.mean(losses)
    return out
```


### The Train Loop

We now show you the most minimal version of the training loop implemented in our code. After initializing the optimizer state, we make a step through every iteration. The loop is adapted to account for resuming stages as well. You may view the logging steps utilized in our project for an additional perspective.

```python
optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

for local_iter_num in range(iter_num, max_iters):
    x, y = get_batch("train")
    
    # do a training step
    model, optimizer_state, loss = make_step(model, optimizer_state, x, y)
```

### Saving the Model

We use the following logic to save the model parameters as well as the training configuration. We once again encourage the reader to refer our repo for the complete implementation of this logic.

```python
checkpoint_params = {
    "model_args": gptconf,
    "iter_num": local_iter_num,
    "val_loss": losses["val"],
    "learning_rate": lr,
    "config": config,
}

checkpoint_file = os.path.join(out_dir, 'model.eqx')
checkpoint_params_file = os.path.join(out_dir, 'params.pkl')

eqx.tree_serialise_leaves(checkpoint_file, model)

with open(checkpoint_params_file, "wb") as f:
    cloudpickle.dump(checkpoint_params, f)
```

<hr />

## Conclusion

If you've reached this far through the sections, congratulations on your dedication to exploring _JAX_ and _Equinox_! In this blog post, we've taken a unique approach to learning these powerful frameworks by rewriting the well-known [nanoGPT](https://github.com/karpathy/nanoGPT) repository step by step.

Throughout this process, we've encountered and overcome several challenges unique to _JAX_'s immutable nature and _PyTree_ definition. From reimagining the model architecture to adapting the training loop, each step helped us learn how to effectively leverage _JAX_ and _Equinox_ for complex deep learning tasks. We saw how to:

1. Implement custom initializations.
2. Handle model parameters as _PyTrees_.
3. Use _Equinox_'s filtered transformations like `equinox.filter_jit` and `equinox.filter_grad` to work with non-array objects in our model.

We've explored _JAX_'s transformations, particularly `vmap`, to create efficient, parallelized code for handling batched inputs across various layers of our model. _Equinox_'s ability to seamlessly integrate with _JAX_ while providing a familiar _PyTorch_-like interface for building neural networks proved invaluable. Notably, _Equinox_'s filtered transformations were crucial in applying _JAX_'s powerful _JIT_ compilation and automatic differentiation to our model, as we saw in the `compute_loss` and `make_step` functions.

This rewrite not only serves as a learning exercise but also demonstrates the flexibility and power of _JAX_ and _Equinox_ in handling complex deep learning models. By working through this example, we hope you've gained a deeper understanding of these frameworks and feel more confident in applying them to your own projects.

As we conclude, remember that this is just the beginning. The field of machine learning is constantly evolving, and frameworks like _JAX_ and _Equinox_ are only a pitstop in a never ending journey. We encourage you to continue exploring, experimenting, and pushing the boundaries of what's possible with these tools and more.

For those inspired to dive deeper, the entire codebase for this project is open-sourced and can be found [https://github.com/surgeglobal/nanoJAXGPT](https://github.com/surgeglobal/nanoJAXGPT). We hope this resource serves as a springboard for your own explorations in _JAX_ and _Equinox_. May your journey in machine learning be filled with exciting discoveries and groundbreaking innovations!

<div style="display: flex; align-items: flex-start; background-color: #1e1e2e; padding: 16px; border-radius: 8px; font-family: Arial, sans-serif; color: white;">
    <div style="margin-right: 12px;">
        <img src="https://img.icons8.com/?size=100&id=48250&format=png&color=000000" alt="repo-icon" style="width: 40px; height: 40px; margin: 0; background-color: transparent; border: none;">
    </div>
    <div>
        <h3 style="margin: 0; font-size: 15px;"><a href="https://github.com/surgeglobal/nanoJAXGPT" style="text-decoration: none; color: #7aa2f7;">surgeglobal/nanoJAXGPT</a></h3>
        <p style="margin: 4px 0; font-size: 14px; color: #a9b1d6;">Created by Surge Global • Updated on Jun 6, 2024</p>
    </div>
</div>


## Acknowledgements
* We thank [Andrej Karpathy](https://karpathy.ai/) for his elegent repository of _nanoGPT_ which has helped us understand the _GPT_ architecture and contribute with a _JAX/Equinox_ version of their project.
* We are also grateful for [Anh Tong](https://github.com/anh-tong) whose _Equinox_ version of _nanoGPT_ was a source of inspiration for our unique rewrite. We recommend referring to his version of nanoGPT as well here: [https://github.com/anh-tong/nanoGPT-equinox](https://github.com/anh-tong/nanoGPT-equinox).
* The [JAX](https://jax.readthedocs.io/en/latest/index.html) team for an amazing framework.
* The [Equinox](https://docs.kidger.site/equinox/) team for making JAX feel like PyTorch.
* The [Modal](https://modal.com/) team for their effort in making serverless GPU usage accessible and affordable. Most importantly, for providing a free $30 credit for each workspace in your account.
* This blogpost is powered by free icons from [Icons8](https://icons8.com).
  * <a target="_blank" href="https://icons8.com/icon/VQOfeAx5KWTK/info">Info</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>
  * <a target="_blank" href="https://icons8.com/icon/hP6pCUyT8QGk/error">Warning</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>
  * <a target="_blank" href="https://icons8.com/icon/48250/code">Code</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>