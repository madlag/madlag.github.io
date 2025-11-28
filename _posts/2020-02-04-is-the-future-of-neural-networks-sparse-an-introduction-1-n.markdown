---
layout:	post
date:	2020-02-04
title:	"Is the future of Neural Networks Sparse? An Introduction (1/N)"
description: "From principles to real-world library support."
author: francois
tags: [ AI, NLP]
categories: []
image: assets/images/posts/sparse1/sparse_1_cover.png
featured: true
---
 
**Hi, I am François Lagunas.**

I am doing Machine Learning research, and I have been working for the last months on using sparse matrices, especially in Transformers.

<div style="clear: both;"></div>
 The recent [**announcement**](https://openai.com/blog/openai-pytorch/) that **OpenAI** is porting its [**block sparse toolbox**](https://openai.com/blog/block-sparse-gpu-kernels/) in **PyTorch** is really big news:


> “We are in the process of writing PyTorch bindings for our highly-optimized blocksparse kernels, and will open-source those bindings in upcoming months.”

I was talking about it with the outstanding [Hugging Face](https://huggingface.co/) team, (I am one of their early investors), and I wanted to share with you my excitement!

### What is a Sparse Matrix?

A ***sparse*** matrix is just a matrix with some zeros. Usually, a lot of them. So every place you are using a ***dense matrix***, in a linear layer, for example, you could be using a sparse one.

<figure class="figcenter">
<img class="large" alt="Matrices with increasing sparsity" src="/assets/images/posts/sparse1/sparse.png">
<figcaption>Matrices with increasing sparsity</figcaption>
</figure>


The ***sparsity*** of the matrix is the fraction of zeros against the size of the matrix

**The pros?** If you have a lot of zeros, you don’t have to compute some multiplications, and you don’t have to store them. So you ***may*** gain on size and speed, for training and inference (more on this today).

**The cons?** Of course, having all these zeros will probably have an impact on network accuracy/performance. But to what extent? You may be surprised.

### Where are they from?

The first researchers/engineers to use sparse matrices were [Finite Elements](https://en.wikipedia.org/wiki/Finite_element_method) users.


<figure class="figcenter">
<img class="large" alt="A 2D mesh (roof of Omni Coliseum, Atlanta) and its finite element matrix" src="/assets/images/posts/sparse1/finite_elements_composite2.png">
<figcaption>A <a href="https://www.cise.ufl.edu/research/sparse/matrices/HB/bcsstk14.html">2D mesh</a> (roof of Omni Coliseum, Atlanta) and its finite element matrix</figcaption>
</figure>


When you have to deal with large physical simulations, you get a large graph of interconnected vertices.

Each vertex is a point of your system, and each edge connects two vertices. That means that these **two points** will have some **influence** on each other in the model. And so there is a **non-zero** value in the matrix that describes the graph.

This last sentence sums it up: you need non-zero values in the matrix when two dimensions are interacting in some way.

**Now getting back to ML, you should ask yourself the same question: are all the dimensions of my input vector interacting with all the others? **Usually not. So going sparse maybe useful.

We have actually a very good, and famous, example of a successful trip to sparse-land: **convolutional layers**.

<figure class="figcenter">
<img class="large" alt="Learned convolutional filters" src="/assets/images/posts/sparse1/convolutions.jpeg">
<figcaption>Learned convolutional filters (from <a href="http://cs231n.github.io/convolutional-networks">http://cs231n.github.io/convolutional-networks</a>)</figcaption>
</figure>


Convolutional layers are a smart and efficient way to implement a sparse transformation on an input tensor.

When processing images, it comes down to two things:

**Sparsity**: the transformation is local → each output pixel should depend on a few neighboring input pixels.

**Invariance**: the transformation does not depend on the position in the image

Then you just add the constraint that the transformation is linear: if you were to represent this transformation, you would get a HUGE matrix with only a few non-zeros. But of course, the right way to do this is to do a multiplication of the input tensor with a small set of small matrices (each square in the image before).

The importance of convolutions in today’s ML success is obvious. But you can see that **finding a clever way to make things sparse sounds like a good recipe to save time and space.**

### Where are they useful?

Convolutions are already an efficient form of sparsity, so you could try to make them [even](https://arxiv.org/abs/1902.05967) more [sparse](http://arxiv.org/abs/1907.04840), but some other networks contain much larger matrices that may benefit from sparsity: Transformers.

And those are getting bigger and bigger. We have greatly exceeded the 1 billion parameters in 2019, and it’s not stopping here. The cost to train and to use those networks is getting unpractical, so every method to reduce their size will be welcome.

<figure class="figcenter">
<img alt="From Nvidia blog" src="/assets/images/posts/sparse1/Nvidia-Blog-Figure-1-Training.jpg">
<figcaption style="margin-top:20px">From <a href="https://devblogs.nvidia.com/training-bert-with-gpus">https://devblogs.nvidia.com/training-bert-with-gpus</a>)</figcaption>
</figure>

### Why the OpenAI announcement is so important?

So, if everything is fine in sparse-land, we should all be trying sparse matrices, shouldn’t we?

Yes. But there is this stupid thing called **implementation**. It’s easy to see the theoretical improvements we could get with sparse compute. But the support in libraries is quite … sparse.

PyTorch [developers](https://github.com/soumith), for example, have done a **significant** **effort** to support sparse compute. But there is still a big gap in performance between dense and sparse matrices operations, which defeats the whole purpose of using them. Even memory usage is quite large: sparsity has to be more than 80% to save some room on sparse matrices (more on that in my next post). Even basic serialization was broken before version 1.4. The reason is that the underlying libraries (for example cuSPARSE) are not doing a great job because the problem is ill-suited to the way GPU works.

So the **OpenAI** **announcement** on their block sparse tools is **very** **good** **news** for those who want to use sparse ops without sacrificing training speed (and it looks like some [people](https://github.com/openai/blocksparse/issues/2) have been waiting for some time now). And we are not talking about a few percents.


> “Our kernels typically performed **one or two orders of magnitude faster** in terms of GFLOPS.”


<figure class="figcenter">
<img alt="From OpenAI blocksparse paper" src="/assets/images/posts/sparse1/openai_speedup.png">
<figcaption>From OpenAI <a href="https://d4mucfpksywv.cloudfront.net/blocksparse/blocksparsepaper.pdf">blocksparse paper</a></figcaption>
</figure>

The worst thing is that the [paper](https://d4mucfpksywv.cloudfront.net/blocksparse/blocksparsepaper.pdf) concludes that cuBLAS is faster that cuSPARSE even with very sparse matrices. How sad.

The magic keyword here is “**block**”. **It’s hard to implement general sparse matrice computations on GPUs in an efficient way**. But it gets much easier if you add a “reasonable” constraint on the form of the matrices: their non-zeros should be grouped in small fixed-size blocks, and that makes GPU processing much easier to parallelize efficiently. Typically 8x8, 16x16 or 32x32 blocks, 16x16 already giving a very good performance, with 32x32 giving a slightly better one.


<figure class="figcenter">
<img alt="A 8-block-sparse matrix" src="/assets/images/posts/sparse1/block_sparse.png">
<figcaption>A 8-block-sparse matrix</figcaption>
</figure>

Of course, the “block” constraint may be crippling some sparsification algorithms, or at least it would require some changes to take it into account.

But at least we can play with large high sparsity matrices, and the block constraint may not be a big issue: if you think about it, it means that there is **some locality in the dimensions**, and that sounds a quite reasonable constraint. That’s the same reason band matrices have been useful in the past (finite difference, finite elements), and it was a much stronger constraint.


<figure class="figcenter">
<img alt="Band matrix" src="/assets/images/posts/sparse1/band_matrix.png">
<figcaption>Band matrix</figcaption>
</figure>

### Conclusion

I hope I have convinced you that 2020 will be the sparse network year (it already has two zeros, that’s a sign).

**Next time** for those who are curious about what happens when they are using some CUDA based PyTorch code, we’ll dig a bit deeper in **GPU internals**, (and we will understand **why block sparse code is outrunning sparse code by a large margin**).

**This article series will continue on the different techniques that have been proposed to make sparse networks, and what are the potential long term benefits.**

#### More reading

First, here is a **[study](https://towardsdatascience.com/sparse-matrices-in-pytorch-part-2-gpus-fd9cc0725b71) of PyTorch sparse performance.**

If you want to have a very detailed review of **different complementary approaches to network size reduction**, and not just about sparse ones, you should definitely read [this article](http://mitchgordon.me/machine/learning/2020/01/13/do-we-really-need-model-compression.html).

And if you want to **create illustrations like the header of this blog post**, you will find the code I used on my **[github](https://github.com/madlag/medium_posts/tree/master/sparse_matrices_1)**.

  