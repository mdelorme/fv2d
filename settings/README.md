# fv2d numerical tests

You can find in this directory all the tests used for `fv2d`'s validation. 
Parameters used for each tests are given here, but for more details please visit [fvNd-unit-test](https://github.com/lukbrb/fvNd-unit-test).
This companion repository aims to explain the importance of each test, and the results obtained with all `fvNd` codes; for $N \in \left\{1, 2, 3 \right\}$.

## 1D Problems

| Test Case Name | $(\gamma, t_{end}, L)$| $\rho\quad(u, v, w)\quad p\quad(B_x, B_y, B_z)$|
| :--------------|:--------------------------------:| :------------|
|**Dai & Woodward**  |$(\frac{5}{3}, 0.2, 1.1)$         |               |
| Left State ||$1.080 \quad (1.2, 0.01, 0.5)\quad 0.95\quad\frac{1}{\sqrt{4}}(4.0,3.6, 2.0)$
| Right State ||$1.000 \quad (0.00, 0.0, 0.0) \quad 1.00 \quad \frac{1}{\sqrt{4}}(4.0, 4.0, 2.0)$
| **Brio & Wu I**| $(2.0, 0.2, 1.0)$||
| Left State ||$1.000 \quad (0.0, 0.0, 0.0) \quad 1.0 \quad(0.65, +1.0, 0.0)$
| Right State ||$0.125 \quad (0.0, 0.0, 0.0) \quad 0.1 \quad(0.65, -1.0, 0.0)$
| **Brio & Wu II**| $(2.0, 0.012, 1.4)$||
| Left State ||$1.000 \quad (0.0, 0.0, 0.0) \quad 1000 \quad(0.0, +1.0, 0.0)$
| Right State ||$0.125 \quad (0.0, 0.0, 0.0) \quad 0.100 \quad(0.0, -1.0, 0.0)$
| **Slow Rarefaction**| $(\frac{5}{3}, 0.2, 1.0)$||
| Left State ||$1.0 \quad (0.000, 0.000, 0.000) \quad 2.0000 \quad(1.0, 0.0000, 0.0)$
| Right State ||$0.2 \quad (1.186, 2.967, 0.000) \quad 0.1368 \quad(1.0, 1.6405, 0.0)$
| **Expansion I**| $(\frac{5}{3}, 0.15, 1.4)$||
| Left State ||$1.0 \quad (-3.1, 0.0, 0.0) \quad 0.45 \quad(0.0, 0.5, 0.0)$
| Right State ||$1.0 \quad(+3.1, 0.0, 0.0) \quad 0.45 \quad(0.0, 0.5, 0.0)$
| **Expansion II**| $(\frac{5}{3}, 0.15, 1.4)$||
| Left State ||$1.0 \quad (-3.1, 0.0, 0.0) \quad 0.45 \quad(1.0, 0.5, 0.0)$
| Right State ||$1.0 \quad (-3.1, 0.0, 0.0) \quad 0.45 \quad(1.0, 0.5, 0.0)$

## 2D Problems
