# PID-controlled Langevin Dynamics for Faster Sampling of Generative Models

This is the official implementation of the paper "PID-controlled Langevin Dynamics for Faster Sampling of Generative Models". Each directory corresponds to a task. Please refer to the `README.md` file in each directory for more implementation details.

[Paper](https://openreview.net/forum?id=y9LHDCKeeN) · [NeurIPS](https://neurips.cc/virtual/2025/loc/san-diego/poster/115179) · [Code](https://github.com/tsinghua-fib-lab/PIDLD) · [WebPage](https://tsinghua-fib-lab.github.io/PIDLD/)


## 🔍 Highlights

![PID illustration](fig/intro_4.jpg)

- **Control-theoretic insight:** Reinterprets Langevin dynamics as a feedback control system, where energy gradients act as feedback signals.  
- **PID-enhanced sampling:** Integrates Proportional, Integral, and Derivative control terms into Langevin updates:
  - **P-term:** basic gradient guidance;
  - **I-term:** accumulates historical gradients for momentum-like acceleration;
  - **D-term:** anticipates gradient trends for adaptive stabilization.
- **Plug-and-play compatibility:** Requires no retraining or prior information; integrates with any Langevin-based sampler directly (EBM, SGM, etc.).
- **Significant speedup:** Achieves up to **10× faster sampling** while maintaining or improving generation quality across image and reasoning tasks.


## ⚙️ Algorithm Workflow

The PID-controlled Langevin dynamics update is given by

$$\begin{aligned}
x_{t+1}=x_t+&\epsilon\Big(\\
&\quad\ k_p\nabla_{x}U_\theta(x_t)\\
&+k_i\cdot\frac{1}{t}\sum_{s=0}^{t}\nabla_{x}U_\theta(x_s)\\
&+k_d(\nabla_{x}U_\theta(x_t)-\nabla_xU_\theta(x_{t-1}))\\
&\Big)+\sqrt{2\epsilon}\xi_t,\end{aligned}$$

where $k_p,k_i,k_d$ are the proportional, integral, and derivative gains, $U_{\theta}(\cdot)$ is the energy function, $\epsilon$ is the learning rate, and $\xi_t\sim\mathcal{N}(0,I)$.

**PIDLD Algorithm Flowchart**

1. **Require:** Score function $\nabla_x U_\theta(x)=\nabla_x\log p_\theta(x)=\nabla_x(-f_\theta(x))$; number of steps $T$; step size $\epsilon$; control parameters $k_p,k_i,k_d$; decay rate $\gamma<1$; initial point $x_0$.
2. Initialize integral term $I_0 = 0$.
3. Compute initial score $s_0 = \nabla_x U_\theta(x_0)$.
4. For $t = 0$ to $T-1$ do:
  1. $s_t = \nabla_x U_\theta(x_t)$
  2. $P_t = s_t$  (Proportional term)
  3. $I_t = \dfrac{1}{t+1}\big(I_{t-1}\cdot t + s_t\big)$  (Integral term)
  4. $D_t = s_t - s_{t-1}$  (Derivative term)
  5. $u_t = k_p P_t + k_i I_t + k_d D_t$  (Control signal)
  6. State update:
      $$x_{t+1} = x_t + \epsilon \cdot u_t + \sqrt{2\epsilon}\xi_t,\quad \xi_t \sim \mathcal{N}(0,I)$$
  7. Decay integral gain: $k_i = k_i \cdot \gamma$
5. End for
6. **Return:** $\hat{x} = x_T$


## 📊 Experiments

We evaluate PIDLD against standard Langevin-based samplers (vanilla ALD and MILD) across three regimes: toy 2‑D examples, image generation, and reasoning (solution sampling). The focus is on sampling quality versus computational budget (NFE).

- **Toy experiments**
  - Purpose: validate the roles of P/I/D terms on simple multimodal landscapes.
  - Findings: both I and D terms accelerate convergence and reduce KL/divergence; in particular, D term improves stability, and I term reduces steady-state bias.

- **Image generation (CIFAR10, CelebA)**
  - Setup: apply PIDLD as a plug‑in to pretrained score-based models (NCSNv2) and energy models (IGEBM), vary NFE and tune PID gains (with decaying k_i); compute FID on 10k samples.
  - Goal: test whether PIDLD can reach baseline or better image quality at substantially lower NFEs.
  - Results: PIDLD consistently matches or outperforms baselines at much lower NFEs, demonstrating clear efficiency gains.

- **Reasoning (Sudoku, Connectivity)**
  - Setup: use an energy-based solver (IRED) and evaluate solution accuracy under different NFEs.
  - Goal: assess whether PIDLD helps navigate complex energy landscapes to find valid solutions faster.
  - Results: PIDLD yields higher accuracy with lower NFEs, showing overall computational advantages versus vanilla Langevin dynamics sampling.

For more details, please refer to the paper and the code.


## 📚 Citation

If you find the idea useful for your research, please consider citing:

```bibtex
@inproceedings{chen2025pidcontrolled,
  title={{PID}-controlled Langevin Dynamics for Faster Sampling on Generative Models},
  author={Hongyi Chen and Jianhai Shu and Jingtao Ding and Yong Li and Xiao-Ping Zhang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=y9LHDCKeeN},
}
```
