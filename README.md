## CNN Introduction

### What are Convolutional Neural Networks (CNNs)
---
Convolutional neural networks (CNNs) are a subset of deep learning neural networks, which themselves fall under machine learning, focusing on learning hierarchical feature representations from data. They are specifically designed to handle to two-dimensional data, such as images, by exploiting special localit. CNNs are commonly used in computer vision, enabling computers to interpret visual data, such as recongnizing objects in images or videos.

The architecture of CNNs is inspired by the visual perception. A biological neuron corresponds to an artificial neuron; CNN kernels represent different receptors that can respond to various features; activation functions simulate the function that only neuron electric signals exceeding a certain threshold can be transmitted to the next neuron. Loss functions and optimization are something people invented to teach the whole CNN system to learn what we expected.

### Advantages of CNNs
---
Compared with general artificial neurons, CNN possesses many advantages:
1. **Local Connections:** Each neuron is no longer connected to all neurons of the previous layer, but only to a small number of neurons, which is effective in reducing the parameters and speed up convergence.
2. **Weight Sharing:** A group of connections can share the same weights, which reduces parameters even furthur.
3. **Downsampling Dimension Reduction:** A pooling layer harness the principle of image local correlations to downsample an image, which can reduce the amount of data while retaining useful information. It can also reduce the number of parameters by removing trivial features.
4. **Hierarchical Feature Learning:**  Lower layers often learn simple features (e.g., edges), while deeper layers can learn more complex patterns (e.g., object parts), enabling robust representations for high-level tasks.

### Basic CNN Components
---
![](imgs/image.png) 
> Image source: [analyticsvidhya](https://www.analyticsvidhya.com/blog/2022/03/basics-of-cnn-in-deep-learning/)

Many CNN models follow a fixed structure: an input layer, alternating convolution and pooling layers that form a feature extractor, fully connected layers with activation functions, and an output layer. The feature extractor converts raw input into higher level representations, which are then used for tasks like classification or regression. Additional componentssuch as batch normalization and dropout furthur enhance performance.

1. **Convolutional Layers:** Convolutional layers are the foundation of CNNs, designed to extract features from input data, such as edges, textures, or shapes in images. They use filters (kernels), which are small matrices (e.g., 3x3 or 5x5) that slide over the input data, performing a convolution operation. This operation involves computing the dot product between the filter and a local region of the input, producing a feature map that highlights the presence of specific patterns.
2. **Pooling Layers:** Pooling layers follow convolutional layers and reduce the spatial dimensions (width and height) of feature maps, making the network more computationally efficient and less prone to overfitting. Pooling acts like a summarization step, retaining important information while discarding less critical details.
3. **Activation Functions:** Activation functions introduce non-linearity into the network, allowing it to learn complex patterns that linear operations alone cannot capture. Without non-linearity, a deep network would behave like a single linear transformation, limiting its power. The most popular activation function in CNNs is ReLU, defined as $f(x)=max(0, x)$.
4. **Fully connected Layers:** Fully connected layers are typically placed at the end of a CNN and are responsible for making predictions, such as classifying an image into categories (e.g., "cat" or "dog").
5. **Batch Normalization:** Batch normalization stabilizes and accelerates training by normalizing the inputs to each layer within a mini-batch. It is typically applied before the activation function in convolutional or fully connected layers.
6. **Dropout:** Dropout is a regularization technique that prevents overfitting by randomly deactivating neurons during training.

**References:**
- Z. Li, F. Liu, W. Yang, S. Peng and J. Zhou, "[A Survey of Convolutional Neural Networks: Analysis, Applications, and Prospects](https://ieeexplore.ieee.org/document/9451544)," in IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 12, pp. 6999-7019, Dec. 2022, doi: 10.1109/TNNLS.2021.3084827. 
- Zhao, X., Wang, L., Zhang, Y. et al. [A review of convolutional neural networks in computer vision](https://doi.org/10.1007/s10462-024-10721-6). Artif Intell Rev 57, 99 (2024). 

## Modern CNN architectures

We will explore key CNN architectures and train them on a resized version of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). The list is organized chronologically and is not exhaustive. 

1. [**AlexNet (Krizhevsky et al., 2012):**](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) The first large-scale network deployed to beat conventional computer vision methods on a large-scale vision challenge, leveraging ReLU activations and dropout for improved performance. ([notebook](./01_alexnet.ipynb))
2. [**Network in Network (NiN) (Lin et al., 2013):**](https://arxiv.org/abs/1312.4400) A pioneering approach that convolves whole neural networks patch-wise over inputs, introducing mlpconv layers to enhance feature abstraction. ([notebook](./02_network_in_network.ipynb))
3. [**VGG Network (Simonyan and Zisserman, 2014):**](https://arxiv.org/abs/1409.1556) A model that makes use of a number of repeating blocks of elements, known for its simplicity and deep stacks of small 3x3 convolution filters. ([notebook](./03_vgg.ipynb))
4. [**GoogLeNet (Szegedy et al., 2015):**](https://arxiv.org/abs/1409.4842) An architecture that uses networks with multi-branch convolutions, introducing the Inception module to capture features at multiple scales efficiently.
5. [**Residual Network (ResNet) (He et al., 2016):**](https://arxiv.org/abs/1512.03385) A widely adopted framework that remains one of the most popular off-the-shelf architectures in computer vision, utilizing residual connections to enable training of very deep networks.
6. [**ResNeXt Blocks (Xie et al., 2017):**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf) An advancement for sparser connections, extending ResNet with grouped convolutions to enhance efficiency and performance.
7. [**DenseNet (Huang et al., 2017):**](https://arxiv.org/abs/1608.06993) A generalization of the residual architecture, where each layer connects to every previous layer, promoting feature reuse and parameter efficiency.
8. [**MobileNet (Howard et al., 2017):**](https://arxiv.org/abs/1704.04861) A lightweight network designed for mobile and embedded applications, employing depthwise separable convolutions to reduce computational complexity.
9. [**EfficientNet (Tan and Le, 2019):**](https://arxiv.org/abs/1905.11946) A state-of-the-art model that scales networks uniformly across depth, width, and resolution, achieving top performance with fewer parameters.

Code Reference:
- [Dive Into Deep Learning - Zhang et al.](https://d2l.ai/)