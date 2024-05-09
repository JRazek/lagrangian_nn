# Lagrangian NN
Project of a lagrangian nn inspired by a paper [Lagrangian Neural Networks](https://arxiv.org/pdf/2003.04630).

## Problem
Consider a system of a harmonic oscillator, described by equations:
$$q(t) = A \cos(\omega t + \phi)$$.
$$\dot{q}(t) = -A \omega \sin(\omega t + \phi)$$.
$$\ddot{q}(t) = -A \omega^2 \cos(\omega t + \phi)$$.

Lagrangian network has a goal to learn a real function $L(q, \dot{q})$ that approximates the lagrangian of that system. In general one would want to learn such lagrangian without EOMs, given only the data points $$\{q_i \dot{q}_i, \ddot{q}_i\}$$.

## Approach
To train such network, one needs a way to calculate loss of such network. Suppose, $(q(t), \dot{q}(t))$ are coordinates in phase space, parametrized by $t$. Then, Euler-Lagrange equations is:
$$\frac{d}{dt} \frac{\partial L}{\partial \dot{q}} - \frac{\partial L}{\partial q} = 0$$.
Applying chain rule and rearranging, it follows:
$$\ddot{q} = H^{-1} (\frac{\partial L}{\partial{q}} - [\frac{\partial L}{\partial {q_i} \partial{\dot q_j}} ] \dot{q} )$$, where $H$ is a Hessian matrix.

