# Compressed Communication for Large-scale Distributed Deep Learning --- A Tutorial

## Tutorial Venue 
[IJCAI 2020](https://ijcai20.org/), Yokohoma, Japan

## Tutorial Dates 
11-13th July, 2020 

## Presenters 
El Houcine Bergou, <houcine.bergou@kaust.edu.sa>

Aritra Dutta, <aritra.dutta@kaust.edu.sa>, [Personal Website](https://www.aritradutta.com/)

Panos Kalnis, <panos.kalnis@kaust.edu.sa>, [Personal Website](https://cloud.kaust.edu.sa/Pages/Kalnis.aspx)

[King Abdullah University of Science and Technology (KAUST)](https://www.kaust.edu.sa/en)


## Description
We survey compressed communication methods for distributed deep learning and discuss the theoretical background, as well as practical deployment on TF and PT. We also present quantitative comparison of the training speed and model accuracy of compressed communication methods on popular deep neural network models and datasets.

## Abstract 
Recent advances in machine learning and availability of huge corpus of digital data resulted in an explosive growth of DNN model sizes; consequently, the required computational resources have dramatically increased. As a result, distributed learning is becoming the de-facto norm. However, scaling various systems to support fast DNN training on large clusters of compute nodes, is  challenging. Recent works have identified that most distributed training workloads are *communication-bound*. To remedy the network bottleneck, various compression techniques emerged, including sparsification and quantization of the communicated gradients, as well as low-rank methods. Despite the potential gains, researchers and practitioners face a daunting task when choosing an appropriate compression technique. The reason is that training speed and model accuracy depend on multiple factors such as the actual framework used for the implementation, the communication library, the network bandwidth and the characteristics of the model, to name a few. 

In this tutorial, we will provide an overview of the state-of-the-art gradient compression methods for distributed deep learning. We will present the theoretical background and convergence guaranties of the most representative sparcification, quantization and low-rank compression methods. We will also discuss their practical implementation on {\TF} and {\PT} with different communication libraries, such as Horovod, OpenMPI and NCCL. Additionally, we will present a quantitative comparison of the most popular gradient compression techniques in terms of training speed and model accuracy, for a variety of deep neural network models and datasets. We aim to provide a comprehensive theoretical and practical background that will allow researchers and practitioners to utilize the appropriate compression methods in their projects. 


## Outline of the tutorial

The tutorial is divided into several parts:

* **Part-1** [45min]. Motivation, history, and examples of compression methods: We will present an overview of the state-of-the-art in sparsification, quantization and low-rank methods used for gradient compression. We will describe in details influential methods for sparsification, quantization, and low-rank methods. 

    * We explain their theoretical guaranties in terms of complexity (i.e., convergence speed) and expected error. 
  
* **Part-2** [35min]. Practical implementation: We will discuss the implementation of these techniques and the challenges one can expect therein. We will  present programming APIs on {\TF} and {\PT} that expose the necessary functions for implementing a wide variety of compressed communication methods. We will explain the various bottlenecks of practical implementations, such as the overheads of the compression/decompression algorithms and the effect of the underlying communication libraries such as Horovod, OpenMPI and NCCL. 

* **Part-3** [20min]. We will present a quantitative comparison of different methods on a variety of deep neural network models and across different datasets. We will demonstrate how parameters such as the number of compute nodes, network bandwidth, size and type of the model affect the training speed (in terms of throughput and actual wall-time) and training accuracy. We will also discuss the effect of acceleration techniques (e.g., momentum) and error correction methods (e.g., memory). 
    
 * **Part-4** [5min]. Interactive session with the audience. 

### What is distributed training? 

A distributed optimization problem minimizes a function 
<img  src ="http://tex.s2cms.ru/svg/%24%24%5Cmin_%7Bx%5Cin%20%5CR%5Ed%7D%20f(x)%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5En%20f_i(x)%24%24," alt= "\min_{x\in \R^d} f(x)= \frac{1}{n}\sum_{i=1}^n f_i(x)," />
where n is the number of workers. Each worker has a local copy of the model and has access to a partition of the training data. The workers jointly update the model parameters <img src="http://tex.s2cms.ru/svg/x%5Cin%5Cmathbb%7BR%7D%5Ed" alt = "x\in\mathbb{R}^d"/>, where d corresponds to the number of parameters. Typically, <img src="http://tex.s2cms.ru/svg/f_i" alt = "f_i"/> is an expectation function defined on a random variable that samples the data. This is also known as *distributed data-parallel training* as well. The following figure shows two instances of distributed training. 


[(a) Centralized distributed SGD setup by using n *workers * and unique * master/parameter * server. (b) An example of decentralized distributed SGD by using n machines forming a ring topology.]<img src="Images/Distributed.png"> 

### How distributed training is performed? 

[(a) DNN architecture at node i. (b) Gradient compression mechanism for one of the layer of a DNN.]<img src="Images/DNN.png"> 

### What is the bottleneck? What is the remedy? 

This tutorial focuses on *gradient compression*. 
      
      * Let g_k^{i,L} be the local gradient in worker $i$ at level $L$ of the DNN during training iteration $k$. 
      * Instead of transmitting $g_k^{i,L}$, the worker sends $Q(g_k^{i,L})$, where $Q$ is a compression operator. 
      * The receiver has a decompression operator $Q^{-1}$ that reconstructs the gradient. 

### What is ***Compression***?

We identify four main classes of compressors in the literature---
* **Quantization**---which reduces the number of bits of each element in the gradient tensor,
[**IEEE 754** datastructures: 32 bits, 16bits, and 8 bits flexpoint] <img src="Images/ieee.png">  
* **Sparsification**---which transmits only a few elements per tensor,
[An example of Top-k sparsification]<img src="Images/topk.png">
* **Hybrid methods**---which combine quantization with sparsification, and 
* **Low-rank methods**---which decompose the gradient into low-rank matrices.

<img src="Images/Table.png">



### Is Layer-wise compression better than the full model compression? 

[(a) Layer-wise training and (b) entire model training]
<img src="Images/Layerwise.png"> 

We argue this is better in practice in our recent [AAAI 2020 paper](https://www.aritradutta.com/uploads/1/1/8/8/118819584/main.pdf). Additionally, we provide both [layerwise and full-model implementation](https://github.com/sands-lab/layer-wise-aaai20). 

## A Unified Framework

<img src="Images/Framework.png">
