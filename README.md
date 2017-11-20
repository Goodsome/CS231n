# CS231n 课程笔记

## 1. Introduction to Convolutional Neural Networks for Visual Recognition
### 1.1 History
Alexnet in 2012,

## 2. Image Classification

### 2.1 k-Nearest Neighbot 

#### 2.1.1 k-Nearest Neighbor on images never used
- Very slow at test time
- Distance metrics on pixels are not informative
- Curse of dimensionality

### 2.1 Linear Classification

f(x,W) = Wx + b	
x: 32 * 32 * 3 = 3073	
W: 10 * 3072	
f: 10 * 1	

## 3. Loss Functions and Optimization

### 3.1 Loss Function

$$L_{i}=\sum_{j\neq y_{i}}max(0,S_{j}-S_{y_{i}})$$

