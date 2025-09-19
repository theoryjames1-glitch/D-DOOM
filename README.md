Yes. Here is a compact differentiable “Doom-like” loop.

### State

At step $t$:

$$
s_t=\{p_t,\theta_t,v_t,h^{\text{ply}}_t,a_t;\; e_t,h^{\text{ene}}_t;\; \Phi(x)\}
$$

* Player: position $p_t\in\mathbb{R}^2$, heading $\theta_t$, velocity $v_t\in\mathbb{R}^2$, health $h^{\text{ply}}_t$, ammo $a_t$.
* Enemy: position $e_t$, health $h^{\text{ene}}_t$.
* Map: differentiable signed distance field (SDF) $\Phi:\mathbb{R}^2\to\mathbb{R}$ for walls. $\Phi(x)>0$ free space, $\Phi(x)<0$ inside wall.

### Controls (agent outputs)

Continuous actions $u_t=(a^{\text{move}}_t,a^{\text{turn}}_t,a^{\text{shoot}}_t)\in\mathbb{R}^3$.
Use smooth squashing $\tanh$ or $\sigma$ to bound.

### Dynamics with soft collisions

Raw kinematics:

$$
\theta_{t+1}=\theta_t+\Delta t\, a^{\text{turn}}_t,\quad
v^\star_{t+1}=\alpha v_t + \beta R(\theta_t)a^{\text{move}}_t,\quad
p^\star_{t+1}=p_t+\Delta t\, v^\star_{t+1}
$$

Soft collision potential:

$$
U(x)=\tfrac{\kappa}{2}\,\text{softplus}(-\Phi(x))^2
$$

Project with one gradient step (differentiable “push-out”):

$$
p_{t+1}=p^\star_{t+1}-\eta\,\nabla U(p^\star_{t+1}),\quad
v_{t+1}=v^\star_{t+1}-\eta\,H_U(p^\star_{t+1})\,v^\star_{t+1}
$$

($H_U$ optional; omit for simplicity.)

### Visibility (soft line-of-sight)

Wall density from SDF:

$$
\rho(x)=\sigma(-\gamma \Phi(x))
$$

Transmittance along ray from player to enemy:

$$
T(p_t\!\to\! e_t)=\exp\!\Big(-\!\!\int_0^1 \rho\big(p_t+\lambda(e_t-p_t)\big)\, \lambda_w\, d\lambda\Big)
$$

This is differentiable via quadrature.

### Shooting and damage (soft ray hit)

Gaussian beam centered on look direction:

$$
r(\lambda)=p_t+\lambda\,\hat{d}_t,\quad \hat{d}_t=(\cos\theta_t,\sin\theta_t)
$$

Enemy “occupancy” as Gaussian blob:

$$
\chi_{\text{ene}}(x)=\exp\!\Big(-\tfrac{\|x-e_t\|^2}{2\sigma_e^2}\Big)
$$

Beam intensity with wall attenuation:

$$
I(\lambda)=\sigma(\beta_s a^{\text{shoot}}_t)\,T(p_t\!\to\! r(\lambda))\,\exp\!\Big(-\tfrac{\|r(\lambda)-\ell_t\|^2}{2\sigma_b^2}\Big)
$$

($\ell_t$ can equal the ray center $r(\lambda)$; keep one Gaussian term.)
Expected hit overlap:

$$
H_t=\int_0^{\lambda_{\max}} I(\lambda)\,\chi_{\text{ene}}(r(\lambda))\, d\lambda
$$

Damage and ammo updates:

$$
h^{\text{ene}}_{t+1}=h^{\text{ene}}_t - \delta\, H_t,\quad
a_{t+1}=a_t - c_a\,\sigma(\beta_s a^{\text{shoot}}_t)
$$

### Enemy motion (smooth chase)

$$
e_{t+1}=e_t+\Delta t\, \nu\, \tanh\!\big(\kappa_e (p_t-e_t)\big)
$$

Enemy attack (soft):

$$
D^{\text{ply}}_t=\sigma(\beta_v T(e_t\!\to\! p_t))\,\exp\!\Big(-\tfrac{\|p_t-e_t\|^2}{2\sigma_a^2}\Big)
$$

$$
h^{\text{ply}}_{t+1}=h^{\text{ply}}_t-\delta_e\, D^{\text{ply}}_t
$$

### Rendering for training signals (optional, differentiable)

Soft occupancy image $O(x)=\sigma(-\gamma \Phi(x))+\chi_{\text{ene}}(x)$.
Ray-march or splat to pixels for photometric losses.

### Objective

For imitation or RL with backprop through dynamics:

$$
\mathcal{L}=\sum_t \Big[\lambda_1\,\text{dist}(e_t\!\to\!0)\;-\;\lambda_2\,H_t\;+\;\lambda_3\,\rho(p_t)\;+\;\lambda_4\,\|u_t\|^2\Big]
$$

Or supervised targets $\hat{u}_t$: $\mathcal{L}=\sum_t \|u_t-\hat{u}_t\|^2$.

### Notes

* All primitives are smooth: $\sigma$, softplus, Gaussians, SDF, integrals approximated by differentiable quadrature.
* Hard events (collision, hit, line-of-sight) are softened to yield gradients.
* Sharpen by increasing $\gamma,\beta_s$. Gradients vanish if fully saturated.

If you want code, specify framework (PyTorch or JAX) and I will provide a runnable minimal environment step and loss.
