# fv2d numerical tests

You can find in this directory all the tests used for `fv2d`'s validation. 
Parameters used for each tests are given here, but for more details please visit [fvNd-unit-test](https://github.com/lukbrb/fvNd-unit-test).
This companion repository aims to explain the importance of each test, and the results obtained with all `fvNd` codes; for $N \in [1, 2, 3]$.

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

Note, $r$ designates the distance from the center of the box.
| Test Case Name | $\left(\gamma, [x_i, x_f], [y_i, y_f]\right)$| $\rho$ |$(u, v, w)$ | $p$ |$(B_x, B_y, B_z)$|
| :-------------------|:--------------------------:| :----------------:|:-------------------:|:----------------:|:-----------------:|
|**Orszag-Tang Vortex**  |$\left(\frac{5}{3}, [0, 1], [0, 1]\right)$|||
| Periodic BC||$\frac{25}{36\pi}$ |$(-\sin(2\pi y), \sin(2\pi x), 0.0)$| $\frac{5}{12\pi}$ |$\frac{1}{\sqrt{4}}(-\sin(2\pi y), \sin(2\pi x), 0.0)$
|**MHD Blast Standard**  |$\left(\frac{5}{3}, [0, 1], [0, 1]\right)$||
| Periodic BC|| $1$ |$(0, 0, 0)$|$10 \quad \text{if}\quad r < r_c $|$(\sqrt{2}, \sqrt{2}, 0)$
|||||$0.1 \quad \text{if} \quad r \geq r_c \quad $|with $r_c=0.1$|
|**MHD Blast - low $\beta$**  |$\left(1.4, [0, 1], [0, 1]\right)$||
| Periodic BC|| $1$|$(0, 0, 0)$|$1000 \quad \text{if}\quad r < r_c$|$(250/\sqrt{2}, 250/\sqrt{2}, 0)$
|||||$0.1 \quad \text{if} \quad r \geq r_c$| with $r_c=0.1$|
| **Rotated Shock Tube** | $\left(2.0, [0, 1], [0, 1]\right)$ | | | | |
| Neumann BC | | $1$ | $\textbf{R}(\theta)\cdot u_0 \quad \text{for}\quad x_\theta<y_\theta$ | $20 \quad \text{for}\quad x_\theta < y_\theta$ | $\textbf{R}(\theta) \frac{5}{\sqrt{4\pi}}(1, 1, 0)$ |
| | | | $-\textbf{R}(\theta)\cdot \textbf{u}_0 \quad \text{elsewhere}$ | $1 \quad \text{elsewhere}$ | |
| with $\theta = \arctan(-2)$ , | $\textbf{R}(\theta)$ is defined below, | $\textbf{u}_0 = (0, 10, 0)$ | and $(x_\theta, y_\theta) = (\tan\theta(x-0.5), y-0.5)$ | | |
|**MHD Rotor**| $\left( 1.4, [0,1], [0,1]\right)$||||
|Periodic BC  || $10\quad\text{for}\quad r < r_0$|$\frac{u_0}{r_0}(0.5-y, x-0.5)\quad\text{for}\quad r < r_0$|$1.0$|$\left(\frac{5}{\sqrt{4\pi}}, 0, 0\right)$|
|||$1+9f$|$\frac{fu_0}{r_0}(0.5-y, x-0.5, 0)$|$\quad\text{for}\quad r_1 <r\leq r_0$| |
|||$1\quad\text{elsewhere}$|$(0, 0, 0) \quad\text{elsewhere}$| with $u_0=2, r_0=0.1, r_1=0.115$| $f = (r_1-r)/(r_1-r_0)$|
|**Field Loop Advection**|$\left( \frac{5}{3}, [-1,1], [-0.5,0.5]\right)$| | | | |
|Periodic BC||$1.0$|$(2.0, 1.0, 0.0)$|$1.0$| $\frac{A_0}{r}\left(-x, y\right)\quad\text{if}\quad r<r_0$|
|with $A_0$ set to $0.001$| and the loop radius to $r_0=0.3$||||$(0, 0)\quad\text{otherwise}$|

For the rotated shocktube :

$${R}(\theta)= \begin{pmatrix} \sin\theta & \cos\theta \\\ \cos\theta & -\sin\theta \end{pmatrix}$$

