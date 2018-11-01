# Model Parameter Estimation by SGD
This is a demonstration of the power of Stochastistic Gradient Descent for solving difficult non-convex optimization problems. Tensorflow
is used for numerical implementation.

This project aims to determine the model parameters (poles and zeros) of AR (auto-regressive) systems in noise and ARMA 
(auto-regressive moving average) systems. To do so it:
* Determines an approximate frequency domain representation of the signal by BURG method.
* Parameterizes the poles and zeros.
* Build a loss fuction (or objective function) for matching the parameterized frequency domain representation to the original one.
* Because this is a highly non-convex optimization problem with local minima and the loss function is intractable, SGD is used to solve the optimiztion problem.

For more info regarding the math see "Math.pdf"

Demo:
* AR system in noise: https://www.youtube.com/watch?v=HT8WLNDcW-I

![](https://i.imgur.com/joxV8Qj.gif)

* ARMA system: https://www.youtube.com/watch?v=TSW2mzV_9lE

![](https://i.imgur.com/n4zAbtA.gif) 
