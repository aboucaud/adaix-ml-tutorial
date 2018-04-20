class: center, middle

# Hands on Neural Networks

Emille Ishida - Alexandre Boucaud

---

## Foreword

The following slides provide examples of neural network models written in Python, using the [Keras][keras] library. Keras provides a high level API to create models and to run them with numerical tensor libraries (_backends_) such as [TensorFlow][tf], [CNTK][cntk] or [Theano][theano].

All presented models work with Keras version 2.X.

[keras]: https://keras.io/
[tf]: https://www.tensorflow.org/
[cntk]: https://docs.microsoft.com/fr-fr/cognitive-toolkit/
[theano]: http://www.deeplearning.net/software/theano/

---

## Outline

- Neurons - Layers - MLP
- Activation functions
- Backpropagation - forward vs. backward pass
- Optimization - learning rate
- Convolutional Layers
- Pooling, Dropout
- Tour of deep neural nets

---

## Neurons

---
## Anatomy of a neural net

.center[<img src="img/mlp_annotated.jpeg", width="600px;" />]

---
## Multi-layer perceptron (MLP)

.center[<img src="img/mlp.jpeg", width="600px;" />]

---
## Multi-layer perceptron (MLP)

```python
from keras.models import Sequential
from keras.layers import Dense

# initialize model
model = Sequential()

# add layers
model.add(Dense(4, input_dim=3))
model.add(Dense(4))
model.add(Dense(1))
```

--

```python
# print model structure
model.summary()
```

--

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 20
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5
=================================================================
Total params: 41
Trainable params: 41
Non-trainable params: 0
_________________________________________________________________
```

---

## Activation functions



---

## Activation functions

```python

```

---

## Convolutional nets

.center[<img src="img/convolution_gifs/same_padding_no_strides.gif"/>]
.credits[[arXiv:1603.07285](https://arxiv.org/abs/1603.07285)]

---

## Convolutional nets - strides

.left-column[<img src="img/convolution_gifs/same_padding_no_strides.gif" />]
.right-column[<img src="img/convolution_gifs/padding_strides.gif" />]
.credits[[arXiv:1603.07285](https://arxiv.org/abs/1603.07285)]

---
## Convolutional nets - strides 

.left-column[
```python
from keras.model import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(1, (3, 3), 
                 strides=1, 
                 padding='same', 
                 input_shape=(5, 5, 1)))
model.summary()
```

```
_________________________________________
Layer (type)            Output Shape     
=========================================
conv2d (Conv2D)         (None, 5, 5, 1)  
=========================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________
```
] 
.right-column[
<img src="img/convolution_gifs/same_padding_no_strides.gif" />
] 

---
## Convolutional nets - strides

.left-column[
```python
from keras.model import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(1, (3, 3), 
*                strides=2, 
                 padding='same', 
                 input_shape=(5, 5, 1)))
model.summary()
```

```
_________________________________________
Layer (type)            Output Shape     
=========================================
conv2d (Conv2D)         (None, 3, 3, 1)  
=========================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________
```
]
.right-column[ 
<img src="img/convolution_gifs/padding_strides.gif" />
]
 
