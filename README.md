# pytorch-ComplexExponentialSumNeuralNet

# Simple Neural net layer that uses Complex weight -> Exponential -> Sum, following the fact that functions can be approximated with summation of complex exponential weights.

Dependendies: https://github.com/soumickmj/pytorch-complex <Pytorch - Complex>, pytorch


# Usage

` from model.ComplexModel import *
model = SingleApprox(in_var=env.observation_space.shape[0], final_var=env.action_space.n,
                         features=256) # creates in_var -> features -> final simple linear model without Activations.
`

# Problems
Without activation function, the model learns Cartpole, MountainCar pretty well (can pass tests with 480 seconds training). But with double, triple layers, model weight will diverge, leading to NaN, depends on its initial weight. It means there should be other methods to regulate complex weights, which was easy in real numbers.

