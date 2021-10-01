# Quantum state learning with deep neural networks

Quantum state classification and reconstruction with deep neural networks. This repository contains the code to reproduce parts of the results presented in the paper: 


Classification and reconstruction of optical quantum states with deep neural networks

Shahnawaz Ahmed, Carlos Sánchez Muñoz, Franco Nori, and Anton Frisk Kockum
Phys. Rev. Research 3, 033278 – Published 27 September 2021
arXiv: [https://arxiv.org/abs/2012.02185](https://arxiv.org/abs/2012.02185)

## Installation

The code depends on the [qst-cgan](https://github.com/quantshah/qst-cgan.git) package which can be installed by following the instructions on the Github page.

Installing QST-CGAN package will install dependencies such as `tensorflow` and `qutip`. Additional dependencies are `matplotlib`, `scikit-learn`, `scikit-image`, `opencv-python` and `tf_explain`. However, not all parts of the code require these dependencies. 

Finally, install this `qst-nn` package simply by downloading (cloning) this repository and running:

```
python setup.py develop
```
