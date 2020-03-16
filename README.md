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

We identify four main classes of compressors in the literature: 
* **Quantization**---which reduces the number of bits of each element in the gradient tensor 
* **Sparsification**---which transmits only a few elements per tensor (e.g., only the top-$k$ largest elements);
* **Hybrid methods**---which combine quantization with sparsification;
* **Low-rank methods**---which decompose the gradient into low-rank matrices.

<img src="Images/ieee.png"> <img src="Images/topk.png"> 
      

### Is Layer-wise compression better than the full model compression? 

[(a) Layer-wise training and (b) entire model training]
<img src="Images/Layerwise.png"> 

We argue this is better in practice in our recent [AAAI 2020 paper](https://www.aritradutta.com/uploads/1/1/8/8/118819584/main.pdf). Additionally, we provide both [layerwise and full-model implementation](https://github.com/sands-lab/layer-wise-aaai20). 




## A Unified Framework

<img src="http://tex.s2cms.ru/svg/%5Cbegin%7Balgorithm%7D%0A%5Cfootnotesize%0A%5Ccaption%7BDistributed%20Training%20Loop%7D%5Clabel%7Balg_1%7D%0A%5Ctextbf%7BInput%3A%7D%20Number%20of%20nodes%20%24n%24%2C%20learning%20rate%20%24%5Ceta_k%24%2C%20compression%20%24Q%24%20and%20decompression%20%24Q%5E%7B-1%7D%24%20operators%2C%20memory%20compensation%20function%20%24%5Cphi(%5Ccdot)%24%2C%20and%20memory%20update%20function%20%24%5Cpsi(%5Ccdot)%24.%5C%5C%0A%5Ctextbf%7BOutput%3A%7D%20Trained%20model%20%24x%24.%5C%5C%0A%5Cbegin%7Balgorithmic%7D%0A%5CSTATE%20%5Ctextbf%7BOn%7D%20each%20node%20%24i%24%3A%0A%5CSTATE%20%5Ctextbf%7BInitialize%3A%7D%20%24m_0%5Ei%3D%5Cmathbf%7B0%7D%24%20%5CCOMMENT%7Bvector%20of%20zeros%7D%0A%5CFOR%7B%24k%20%3D%200%2C%201%2C%5Cldots%2C%24%7D%0A%5CSTATE%20%5Ctextbf%7BCalculate%7D%20stochastic%20gradient%20%24%7Bg%7D_%7Bk%7D%5Ei%24%0A%5CSTATE%20%24%5Ctilde%7B%7Bg%7D%7D_%7Bk%7D%5Ei%3DQ(%5Cphi(m_k%5Ei%2Cg_%7Bk%7D%5Ei))%24%0A%5CSTATE%20%24%7Bm_%7Bk%2B1%7D%5Ei%7D%3D%5Cpsi(m_k%5Ei%2C%7Bg%7D_%7Bk%7D%5Ei%2C%5Ctilde%7B%7Bg%7D%7D_%7Bk%7D%5Ei)%24%0A%0A%5CIF%7Bcompressor%20uses%20%7B%5Ctt%20Allreduce%7D%7D%0A%5CSTATE%20%24%5Ctilde%7Bg%7D_%7Bk%7D%3D%20%7B%5Ctt%20Allreduce%7D(%5Ctilde%7Bg%7D_%7Bk%7D%5E%7Bi%7D)%24%0A%5CSTATE%20%24%7Bg%7D_%7Bk%7D%20%3D%20Q%5E%7B-1%7D(%5Ctilde%7Bg%7D_%7Bk%7D)%5C%3B%2F%5C%3Bn%24%0A%0A%5CELSIF%7Bcompressor%20uses%20%7B%5Ctt%20Broadcast%7CAllgather%7D%7D%0A%5CSTATE%20%24%5B%5Ctilde%7Bg%7D_%7Bk%7D%5E%7B1%7D%2C%5Ctilde%7Bg%7D_%7Bk%7D%5E%7B2%7D%2C%5Ccdots%2C%5Ctilde%7Bg%7D_%7Bk%7D%5E%7Bn%7D%5D%3D%7B%5Ctt%20Broadcast%7D(%5Ctilde%7Bg%7D_%7Bk%7D%5E%7Bi%7D)%5C%3B%7B%5Ctt%20%7C%7D%5C%3B%7B%5Ctt%20Allgather%7D(%5Ctilde%7Bg%7D_%7Bk%7D%5E%7Bi%7D)%24%0A%5CSTATE%20%24%5B%7Bg%7D_%7Bk%7D%5E%7B1%7D%2C%7Bg%7D_%7Bk%7D%5E%7B2%7D%2C%5Ccdots%2C%7Bg%7D_%7Bk%7D%5E%7Bn%7D%5D%3DQ%5E%7B-1%7D(%5B%5Ctilde%7Bg%7D_%7Bk%7D%5E%7B1%7D%2C%5Ctilde%7Bg%7D_%7Bk%7D%5E%7B2%7D%2C%5Ccdots%2C%5Ctilde%7Bg%7D_%7Bk%7D%5E%7Bn%7D%5D)%24%0A%5CSTATE%20%24%7Bg%7D_%7Bk%7D%20%3D%20Agg(%5B%7Bg%7D_%7Bk%7D%5E%7B1%7D%2C%7Bg%7D_%7Bk%7D%5E%7B2%7D%2C%5Ccdots%2C%7Bg%7D_%7Bk%7D%5E%7Bn%7D%5D)%24%0A%0A%25%20%5CELSIF%7Bcommunication%20is%20%7B%5Ctt%20Push%5C%26Pull%7D%7D%0A%25%20%5CSTATE%20%5Chspace%7B5mm%7D%5Ctextbf%7BOn%7D%20each%20node%20%24i%24%3A%0A%25%20%5CSTATE%20%5Chspace%7B10mm%7D%24%7B%5Ctt%20Push%7D(%5Ctilde%7Bg%7D_%7Bk%7D%5E%7Bi%7D)%24%0A%0A%25%20%5CSTATE%20%5Chspace%7B5mm%7D%5Ctextbf%7BOn%7D%20Master%3A%0A%25%20%5CSTATE%20%5Chspace%7B10mm%7D%24%5B%7Bg%7D_%7Bk%7D%5E%7B1%7D%2C%7Bg%7D_%7Bk%7D%5E%7B2%7D%2C%7Bg%7D_%7Bk%7D%5E%7B3%7D%2C%5Ccdots%5D%3DQ_%7BW%7D%5E%7B-1%7D(%5B%5Ctilde%7Bg%7D_%7Bk%7D%5E%7B1%7D%2C%5Ctilde%7Bg%7D_%7Bk%7D%5E%7B2%7D%2C%5Ctilde%7Bg%7D_%7Bk%7D%5E%7B3%7D%2C%5Ccdots%5D)%24%0A%25%20%5CSTATE%20%5Chspace%7B10mm%7D%24%7Bg%7D_%7Bk%7D%20%3D%20%7B%5Ctt%20Allgather%7D(%5B%7Bg%7D_%7Bk%7D%5E%7B1%7D%2C%7Bg%7D_%7Bk%7D%5E%7B2%7D%2C%7Bg%7D_%7Bk%7D%5E%7B3%7D%2C%5Ccdots%5D)%24%0A%25%20%5CSTATE%20%5Chspace%7B10mm%7D%24%7B%5Ctilde%7Bg%7D_%7Bk%7D%3DQ_%7BM%7D%5Cleft(%5Cphi_%7BM%7D%5Cleft(m_%7Bk%7D%2C%20g_%7Bk%7D%5Cright)%5Cright)%7D%24%0A%25%20%5CSTATE%20%5Chspace%7B10mm%7D%24%7Bm_%7Bk%2B1%7D%3D%5Cpsi_%7BM%7D%5Cleft(m_%7Bk%7D%2C%20g_%7Bk%7D%2C%20%5Ctilde%7Bg%7D_%7Bk%7D%5Cright)%7D%24%0A%0A%25%20%5CSTATE%20%5Chspace%7B5mm%7D%5Ctextbf%7BOn%7D%20each%20node%20%24i%24%3A%0A%25%20%5CSTATE%20%5Chspace%7B10mm%7D%24%5Ctilde%7Bg%7D_%7Bk%7D%3D%20%7B%5Ctt%20Pull%7D(%5Ctilde%7Bg%7D_%7Bk%7D)%24%0A%25%20%5CSTATE%20%5Chspace%7B10mm%7D%24%7Bg%7D_%7Bk%7D%20%3D%20Q_%7BM%7D%5E%7B-1%7D(%5Ctilde%7Bg%7D_%7Bk%7D)%24%0A%25%20%5CSTATE%20%5Chspace%7B10mm%7D%24x_%7Bk%2B1%7D%5E%7Bi%7D%3Dx_%7Bk%7D%5E%7Bi%7D-%5Ceta_%7Bk%7D%7Bg%7D_%7Bk%7D%24%0A%5CENDIF%0A%5CSTATE%20%24x_%7Bk%2B1%7D%5E%7Bi%7D%3Dx_%7Bk%7D%5E%7Bi%7D-%5Ceta_%7Bk%7D%7Bg%7D_%7Bk%7D%24%0A%5CENDFOR%0A%5CRETURN%20%24%7Bx%7D%24%20%5CCOMMENT%7Beach%20node%20has%20the%20same%20view%20of%20the%20model%7D%0A%5Cend%7Balgorithmic%7D%20%20%0A%5Cend%7Balgorithm%7D." alt = "\begin{algorithm}
\footnotesize
\caption{Distributed Training Loop}\label{alg_1}
\textbf{Input:} Number of nodes n, learning rate \eta_k, compression Q and decompression Q^{-1} operators, memory compensation function \phi(\cdot), and memory update function \psi(\cdot).\\
\textbf{Output:} Trained model x.\\
\begin{algorithmic}
\STATE \textbf{On} each node i:
\STATE \textbf{Initialize:} m_0^i=\mathbf{0}
\FOR{k = 0, 1,\ldots,}
\STATE \textbf{Calculate} stochastic gradient {g}_{k}^i
\STATE \tilde{{g}}_{k}^i=Q(\phi(m_k^i,g_{k}^i))
\STATE {m_{k+1}^i}=\psi(m_k^i,{g}_{k}^i,\tilde{{g}}_{k}^i)
\IF{compressor uses {\tt Allreduce}}
\STATE \tilde{g}_{k}= {\tt Allreduce}(\tilde{g}_{k}^{i})
\STATE {g}_{k} = Q^{-1}(\tilde{g}_{k})\;/\;n
\ELSIF{compressor uses {\tt Broadcast|Allgather}}
\STATE [\tilde{g}_{k}^{1},\tilde{g}_{k}^{2},\cdots,\tilde{g}_{k}^{n}]={\tt Broadcast}(\tilde{g}_{k}^{i})\;{\tt |}\;{\tt Allgather}(\tilde{g}_{k}^{i})
\STATE [{g}_{k}^{1},{g}_{k}^{2},\cdots,{g}_{k}^{n}]=Q^{-1}([\tilde{g}_{k}^{1},\tilde{g}_{k}^{2},\cdots,\tilde{g}_{k}^{n}])$
\STATE {g}_{k} = Agg([{g}_{k}^{1},{g}_{k}^{2},\cdots,{g}_{k}^{n}])
\ENDIF
\STATE x_{k+1}^{i}=x_{k}^{i}-\eta_{k}{g}_{k}
\ENDFOR
\RETURN {x} 
\end{algorithmic}  
\end{algorithm}."/>
