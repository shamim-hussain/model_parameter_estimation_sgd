# Model Parameter Estimation by SGD
This is a demonstration of the power of Stochastistic Gradient Descent for solving difficult non-convex optimization problems.

This project aims to determine the model parameters (poles and zeros) of AR (auto-regressive) systems in noise and ARMA 
(auto-regressive moving average) systems. To do so it:
* Determines an approximate frequency domain representation of the signal by BURG method.
* Parameterizes the poles and zeros.
* Build a loss fuction (or objective function) for matching the parameterized frequency domain representation to the original one.
* Because this is a highly non-convex optimization problem with local minima and the loss function is intractable, SGD is used to solve the 
optimiztion problem.
