# Compressed Communication for Large-scale Distributed Deep Learning -- A Tutorial

## Tutorial Venue 
IJCAI 2020, Yokohoma, Japan

## Tutorial Dates 
11-13th July, 2020 

## Presenters 
El Houcine Bergou, <houcine.bergou@kaust.edu.sa>

Aritra Dutta, <aritra.dutta@kaust.edu.sa>, [Personal Website](https://www.aritradutta.com/)

Panos Kalnis, <panos.kalnis@kaust.edu.sa>

King Abdullah University of Science and Technology (KAUST)


## Description
We survey compressed communication methods for distributed deep learning and discuss the theoretical background, as well as practical deployment on TF and PT. We also present quantitative comparison of the training speed and model accuracy of compressed communication methods on popular deep neural network models and datasets.

## Abstract 
Recent advances in machine learning and availability of huge corpus of digital data resulted in an explosive growth of DNN model sizes; consequently, the required computational resources have dramatically increased. As a result, distributed learning is becoming the de-facto norm. However, scaling various systems to support fast DNN training on large clusters of compute nodes, is  challenging. Recent works have identified that most distributed training workloads are *communication-bound*. To remedy the network bottleneck, various compression techniques emerged, including sparsification and quantization of the communicated gradients, as well as low-rank methods. Despite the potential gains, researchers and practitioners face a daunting task when choosing an appropriate compression technique. The reason is that training speed and model accuracy depend on multiple factors such as the actual framework used for the implementation, the communication library, the network bandwidth and the characteristics of the model, to name a few. 

In this tutorial, we will provide an overview of the state-of-the-art gradient compression methods for distributed deep learning. We will present the theoretical background and convergence guaranties of the most representative sparcification, quantization and low-rank compression methods. We will also discuss their practical implementation on {\TF} and {\PT} with different communication libraries, such as Horovod, OpenMPI and NCCL. Additionally, we will present a quantitative comparison of the most popular gradient compression techniques in terms of training speed and model accuracy, for a variety of deep neural network models and datasets. We aim to provide a comprehensive theoretical and practical background that will allow researchers and practitioners to utilize the appropriate compression methods in their projects. 




### Outline of the tutorial

The tutorial is divided into several parts:

* **Part-1** [45min]. Motivation, history, and examples of compression methods: We will present an overview of the state-of-the-art in sparsification, quantization and low-rank methods used for gradient compression. We will describe in details influential methods for sparsification, quantization, and low-rank methods. 

    * We explain their theoretical guaranties in terms of complexity (i.e., convergence speed) and expected error. 
  
* **Part-2** [35min]. Practical implementation: We will discuss the implementation of these techniques and the challenges one can expect therein. We will  present programming APIs on {\TF} and {\PT} that expose the necessary functions for implementing a wide variety of compressed communication methods. We will explain the various bottlenecks of practical implementations, such as the overheads of the compression/decompression algorithms and the effect of the underlying communication libraries such as Horovod, OpenMPI and NCCL. 

* **Part-3** [20min]. We will present a quantitative comparison of different methods on a variety of deep neural network models and across different datasets. We will demonstrate how parameters such as the number of compute nodes, network bandwidth, size and type of the model affect the training speed (in terms of throughput and actual wall-time) and training accuracy. We will also discuss the effect of acceleration techniques (e.g., momentum) and error correction methods (e.g., memory). 
    
 * **Part-4** [5min]. Interactive session with the audience. 

### What is distributed training? 

<img src="Images/DNN.png"> 

### How distributed training is performed? 

<img src="Images/Layerwise.png"> 

We argue this is better in practice in our recent [AAAI 2020 paper](https://www.aritradutta.com/uploads/1/1/8/8/118819584/main.pdf). Additionally, we provide both [layerwise and full-model implementation](https://github.com/sands-lab/layer-wise-aaai20). 


<img src="https://tex.s2cms.ru/svg/x_%7B1%2C2%7D%20%3D%20%7B-b%5Cpm%5Csqrt%7Bb%5E2%20-%204ac%7D%20%5Cover%202a%7D." alt="x_{1,2} = {-b\pm\sqrt{b^2 - 4ac} \over 2a}." />




<img src="https://tex.s2cms.ru/svg/x_%7B1%2C2%7D%20%3D%20%7B-b%5Cpm%5Csqrt%7Bb%5E2%20-%204ac%7D%20%5Cover%202a%7D." alt="\textbf{Input:} Number of nodes $n$, learning rate $\eta_k$, compression $Q$ and decompression $Q^{-1}$ operators, memory compensation function $\phi(\cdot)$, and memory update function $\psi(\cdot)$.\\
\textbf{Output:} Trained model $x$.\\
\begin{algorithmic}[1] 
\STATE \textbf{On} each node $i$:
\STATE \textbf{Initialize:} $m_0^i=\mathbf{0}$ \COMMENT{vector of zeros}
\FOR{$k = 0, 1,\ldots,$}
\STATE \textbf{Calculate} stochastic gradient ${g}_{k}^i$
\STATE $\tilde{{g}}_{k}^i=Q(\phi(m_k^i,g_{k}^i))$
\STATE ${m_{k+1}^i}=\psi(m_k^i,{g}_{k}^i,\tilde{{g}}_{k}^i)$
\IF{compressor uses {\tt Allreduce}}
\STATE $\tilde{g}_{k}= {\tt Allreduce}(\tilde{g}_{k}^{i})$
\STATE ${g}_{k} = Q^{-1}(\tilde{g}_{k})\;/\;n$
\ELSIF{compressor uses {\tt Broadcast|Allgather}}
\STATE $[\tilde{g}_{k}^{1},\tilde{g}_{k}^{2},\cdots,\tilde{g}_{k}^{n}]={\tt Broadcast}(\tilde{g}_{k}^{i})\;{\tt |}\;{\tt Allgather}(\tilde{g}_{k}^{i})$
\STATE $[{g}_{k}^{1},{g}_{k}^{2},\cdots,{g}_{k}^{n}]=Q^{-1}([\tilde{g}_{k}^{1},\tilde{g}_{k}^{2},\cdots,\tilde{g}_{k}^{n}])$
\STATE ${g}_{k} = Agg([{g}_{k}^{1},{g}_{k}^{2},\cdots,{g}_{k}^{n}])$
\ENDIF
\STATE $x_{k+1}^{i}=x_{k}^{i}-\eta_{k}{g}_{k}$
\ENDFOR
\RETURN ${x}$ \COMMENT{each node has the same view of the model}
\end{algorithmic}  
\end{algorithm}" />
