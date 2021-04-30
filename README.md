# ROLLING-DOWN-A-CROWDED-VALLEY-OF-OPTIMIZERS-DEVELOPMENTS-FROM-SGD
# Deep Learning Optimizers 

### About the topic:

In this blog article I have taken the survey paper published by Sebastian Ruder in 2017, _An overview of gradient descent optimization algorithms_ this paper can also be seen on his [website](https://ruder.io/optimizing-gradient-descent/). Also, along with this towards the end a special focus has been on AdaBelief [paper](https://arxiv.org/abs/2010.07468) published in 2020.

This blog is a reinterpretation of his work, some new optimization algorithms which have become quiet popular has been added in the blog to give a complete overview of optimization algorithms till date.

Furthermore, each optimization method have been seen through their original research papers. The shortcomings of each have been mentioned which is addressed in the subsequent topics introducing new optimization techniques. 

In order to have a establish understanding of all optimizers in a connected way, and to keep away from the different notations used in the papers, all the mathematical equations have been made consistent from start to the end following the same notations for each optimization algorithms to establish a good understanding on how. 

Comparison amongst various optimizers have also been made and shown through diagrams and animations in the blog.

This blog article has been emphasized on explaining the key concepts, shortcomings of previous approches and approaches taken to overcome the shortcomings by evaluating each papers along with the survey papers and further present the concepts in an intuitive way giving an original explanation by connecting all the work done so far. For the working code implementation, code for ADAM optimizer has been provided in the end. 

**Note:** - In case of issues with viewing diagrams and visualisations please refer to the same notebook on google colaboratory as the images have been embedded in the original colab notebook [colab notebook link](https://colab.research.google.com/drive/1MWWUgELkVrCuae-g3BCJTc0Qzl0dpXZY#scrollTo=kW4LXj_L1B8_).


## Introduction <a class="anchor" id="intro"></a>

Optimizers are parameters used in machine learning algorithms to update the weights for the loss function in the backpropagation of the neural network. 

Let's look at 2 essential terms used in machine learning and deep learning algorithms using optimizers as their parameters:

**Epochs** - an event or a time marked by an event that begins a new period or development. (dictionary definition). Epoch indicates the number of passes of the entire training dataset a machine learning algorithm has completed. 
Datasets are broken and grouped into batches (when the amount of data is very large). So, each epoch indicates the pass for each batch-size of the dataset.

**Iterations** - indicates the number of times the algorithm's parameters are updated.

Training of a neural network will require many iterations. At each iteration the parameters and the learning rate is updated and a new iteration is performed with new values for all the epochs. 

A typical example of a single iteration of training of a neural network would include the following steps:
1. processing the training dataset batch-size
2. calculating the cost function
3. backpropagation and adjustment of all weighting factors


In a neural network, first we perform forward propagation for the output and based on that we create a loss or error (or cost) function to calculate the error function.

$$\begin{align}
Loss/Error = \frac{1}{2}(y-\hat{y})^{2} → 	MSE
\end{align}$$

When we compute the loss function the aim is to reduce the error which is done using optimizers.

Optimizers update the weights for the function in backpropagation of the neural network. 


## **1. Gradient Descent *$\color{green}{(Cauchy, 1857)}$*:** <a class="anchor" id="GD"></a>
Gradient Descent (GD) is a method to minimize the cost function J() in each step. It iteratively updates the weights and bias trying to reach the global minimum in the cost function. The new weights are calculated as

$$\begin{align}
W_t = W_{t-1} - \eta . \frac{\partial \mathcal{J}(\mathbf{L})}{\partial \mathbf{W}_{t-1}}
\end{align}$$

Gradient descent takes the complete dataset and then updates the weight for the entire dataset. The process would be very slow for a large number of records (say 10 million records). 

This is because the weights and bias updation will take time and ***hence the convergence will happen slowly.*** Although it provides stable convergence and a stable error, it is very slow for big datasets. This makes Gradient descent more resource intensive using very high RAM and computationally expensive. 


![](https://drive.google.com/uc?export=view&id=1nj8LqkBgN8pvg4116oYlDdCLHDd7bWIS)

*Batch gradient descent is sensitive to saddle points, which can lead to premature convergence*

## **2. Stochastic Gradient Descent (SGD) *$\color{royalblue}{(Robinds, and Monro, 1951)}$*:** <a class="anchor" id="SGD"></a>
To overcome the problem of Gradient descent for the entire dataset, SGD takes a single record at a time, performs the forward propagation and the backpropagation for each record. This will lead to performing 1 epoch at a time. So, the number of iterations will be (1 epoch X Number of records) as compared to Gradient Descent where there was only a single iteration for the entire number of records. 

The process will again be very slow as we are taking 1 record at a time. So for each epoch, the SGD optimizer will perform backpropagation calculating the error or cost function and update the weight each time. Again the weight updation will take time  for each epoch and hence again the convergence will happen slowly. 

Stochastic Gradient Descent is less resource intensive using less RAM and less computationally expensive. 

For such a huge number of records the loss function would be: 

$$\begin{align}
SGD_{Loss} = \frac{1}{2} {\sum}_{i=1}^{N}(y_i-\hat{y_i})^{2} 
\end{align}$$

The error or loss surface can be visualised as a set of contours and can be represented as contour plots as shown below.

![](https://drive.google.com/uc?export=view&id=1toM4CS6P1u4fJ_rbGzuXuzq2l8wmcEP6)

## **3. Mini-batch Gradient Descent:** <a class="anchor" id="Mb-GD"></a>
To overcome the problems in SGD, the researchers came up with an approach to take a mini-batch size for every epoch. So the entire dataset is broken into batches, taken as mini-batch size for the epochs. For 1 million records say for 1000 batch-size, the number of iterations will be (Number of records/batch-size) 1000 iterations. 

This allows in moving quickly to the global minimum in the cost function and updating the weights and biases multiple times per epoch now. Most common mini-batch sizes used are 16, 32, 64, 128, 256, 512 and 1024. 

![](https://drive.google.com/uc?export=view&id=1nUJEtvNq7sLSVDNod_SGTxPioNcQA258)

In comparison to SGD in mini-batch GD the convergence will not be as smooth as SGD. This is because every batch-size in mini-batch GD will create a noise which is added upon to the previous batch-size.

Let's just try to draw in a representative graph what we discussed above. We will be enhancing this representation for further discussions. 
It can be noted that the representation just is for the purpose of understanding, it may not be accurately correct.

![](https://drive.google.com/uc?export=view&id=1jr_HoH5WKMJ9Tjb9ZxvcqczjEi1-dLiq)

Also, compared to a batch gradient descent which is very slow mini-batch performs well.

However, a mini-batch gradient descent may help in escaping a shallow local minima, but it often fails when dealing with a situation of deep local minima.

![](https://drive.google.com/uc?export=view&id=1o6Kqd6-Lp--ccTLWa6hN_MKzFDkdHsmf)
***Even a stochastic error surface won’t save us from a deep local minimum.***
*(Source: Buduma, Nikhil, Fundamentals of Deep Learning)*
