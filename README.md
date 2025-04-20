# Randomized Smoothing as an Adversarial Defense Mechanism for Inverse Problems

Bachelor's thesis project as part of the B.Sc. Electrical Engineering and Information Technology program at TUM.
The code provided does not include the dataset used (ImageNet-1000-mini).

### Abstract
Randomized smoothing is a mechanism that can achieve certifiable robustness of neural network-based
classifiers against $ℓ_2$-norm bounded adversarial examples. In this project, we present an approach to 
randomized smoothing for inverse problems to investigate its effectiveness as an adversarial defense 
mechanism in image reconstruction problems. We choose super-resolution as an image reconstruction problem
to implement randomized smoothing and train U-Net models for super-resolution with different levels of
Gaussian noise for randomized smoothing. We also train U-Net models with adversarial training for a
comparative evaluation of the robustness gains yielded by randomized smoothing in super-resolution. Our
findings show that randomized smoothing is an effective adversarial defense in super-resolution and that
it achieves results with better perceived visual quality than adversarial training.
