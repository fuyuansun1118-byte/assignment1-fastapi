# Assignment 4: Diffusion and Energy-Based Models

## Question 1 — Sinusoidal timestep embedding

Use a zero-based, pair-interleaved embedding of even dimension d, with maximum
period M = 10000. For j = 0,...,d/2-1:

\[
\omega_j=M^{-2j/d},\qquad
\mathrm{emb}_{2j}(t)=\sin(t\omega_j),\qquad
\mathrm{emb}_{2j+1}(t)=\cos(t\omega_j).
\]

Thus the i-th dimension is sin(t/M^(i/d)) for even i; for odd i it is
cos(t/M^((i-1)/d)). The sin/cos pair uses the same frequency. Here t is the
integer diffusion step. Sines and cosines map a scalar timestep to a smooth
vector with multiple frequency scales, so the denoiser can distinguish noise
levels. Our implementation uses this explicit convention.

Some libraries concatenate all sin terms then all cos terms; that changes the
ordering, not the frequency values. Other implementations use a different
frequency grid or embed continuous noise variance, including the current course
practical. Those embeddings must not be numerically confused with the
max-period-10000 convention stated here and used for Questions 1–2.

## Question 2 — Dimension 8, t = 1, max period 10000

For d = 8, j = 0,1,2,3 gives:

\[
\omega=[10000^0,10000^{-1/4},10000^{-1/2},10000^{-3/4}]
       =[1,0.1,0.01,0.001].
\]

The angles at t=1 are [1,0.1,0.01,0.001] radians. With the pair-interleaved
ordering from Question 1, the embedding is:

[sin(1), cos(1), sin(0.1), cos(0.1), sin(0.01), cos(0.01), sin(0.001), cos(0.001)]

Approximately:

**[0.841470985, 0.540302306, 0.099833417, 0.995004165,
0.009999833, 0.999950000, 0.000999999833, 0.999999500]**.

If sin-then-cos concatenation is used instead, the same values appear in the order
[sin(1),sin(0.1),sin(0.01),sin(0.001),cos(1),cos(0.1),cos(0.01),cos(0.001)].

## Question 3 — Relation to transformer positional encoding

Both use sinusoidal functions at multiple frequencies to encode a scalar index
into a vector. In a transformer, the index is the position of a token in a
sequence; positional information is usually combined with token representations
so attention can use sequence order. In a diffusion model, the index is the
noise/denoising timestep, shared across an image's spatial locations. Its
embedding conditions the denoising network (for example, by projection into
residual blocks) so the network knows which noise level it must handle.
The key difference is **sequence position versus diffusion noise level**, and
how that information conditions the architecture.

## Question 4 — UNet bottleneck resolution

For stride-2 convolutions that halve each spatial dimension:

64 x 64 → 32 x 32 → 16 x 16 → **8 x 8**.

Equivalently, 64 / 2^3 = 8. This assumes the padding/kernel choices preserve
exact halving, as in kernel 4, stride 2, padding 1. Our CIFAR-10 models use
native 32 x 32 images, so their three-downsample bottleneck is 4 x 4; that does
not change the answer to the question's specified 64 x 64 input.

## Question 5 — UNet output and loss

For the epsilon-prediction parameterization used here, the UNet takes noisy
image x_t and timestep t and predicts the added Gaussian noise
\(\epsilon_\theta(x_t,t)\), with the **same tensor shape as the input image**.
The network does not directly output a class or a scalar energy.

To form a training pair, sample t and epsilon ~ N(0,I), and calculate:

\[
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon.
\]

Compare the predicted noise with the known sampled noise:

\[
L=\mathbb E_{x_0,t,\epsilon}\left[\|\epsilon-
\epsilon_\theta(x_t,t)\|_2^2\right].
\]

The implementation uses mean squared error averaged over the batch, channels
and pixels. Backpropagation updates the UNet parameters. During generation,
the predicted noise can be used to estimate the clean image:

\[
\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t))
         /\sqrt{\bar\alpha_t}.
\]

The DDIM sampler then combines this estimate and the predicted noise at the
next lower noise level. Other diffusion parameterizations can predict x0 or
velocity v; the answer above concerns this assignment's noise-prediction setup.

## Question 6 — Basic gradient calculations

**a)** For y = x² + 3x, dy/dx = 2x + 3. At x = 2:

**dy/dx = 2(2) + 3 = 7**, so the output is `x.grad = tensor([7.])`.

**b)** If x has `requires_grad=False` and no other input requires gradients,
y has no autograd graph. Calling `y.backward()` raises a RuntimeError stating
that the tensor does not require grad and has no grad_fn; `x.grad` remains None.
It does not silently calculate a zero derivative.

**c)** `torch.tensor` defaults to `requires_grad=False`, so gradients are not
tracked by default. Set `requires_grad=True` explicitly for floating-point or
complex tensors when derivatives with respect to them are needed.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x
y.backward()
print(x.grad)  # tensor([7.])
```

Reference: [official torch.tensor documentation](https://docs.pytorch.org/docs/2.15/generated/torch.tensor.html).

## Question 7 — Introduce weights

The function is y = w0*x² + w1*x, evaluated at x=2 and w=[1,3].

**a)** In the supplied code, x requires gradients but w does not. The output is
`x.grad = tensor([7.])` and **`w.grad = None`**. Autograd tracks the path needed
for x but does not populate a gradient for the non-requiring-grad leaf w.
None indicates no stored gradient, not that the derivative is mathematically zero.

**b)** Enable gradient tracking for w:

```python
x = torch.tensor([2.0])
w = torch.tensor([1.0, 3.0], requires_grad=True)
y = w[0]*x**2 + w[1]*x
y.backward()
print("w.grad =", w.grad)  # tensor([4., 2.])
```

The derivatives are dy/dw0 = x² = **4** and dy/dw1 = x = **2**.
If gradients for both x and w are wanted, set `requires_grad=True` on both;
then x.grad is [7] and w.grad is [4,2].

**c)** As in Question 6, `torch.tensor` defaults to `requires_grad=False`.
Creating w without that flag does not enable tracking for w.

Reference: [official torch.tensor documentation](https://docs.pytorch.org/docs/2.15/generated/torch.tensor.html).

## Question 8 — Breaking the graph

`y = x*3` is connected to x, but `z = y.detach()` removes that autograd
connection. Consequently, w=z*2 has no grad_fn in the supplied example,
and `w.backward()` raises a RuntimeError.

Keep z connected to y by using `clone()` without `detach()`:

```python
x = torch.tensor([1.0], requires_grad=True)
y = x*3
z = y.clone()
w = z*2
w.backward()
print(x.grad)  # tensor([6.])
```

The chain rule gives dw/dx = 2*3 = **6**. Merely setting
`z.requires_grad_(True)` after detaching would create a new leaf; it would
allow z.grad but would **not reconnect the original path to x**.

Reference: [official Tensor.detach documentation](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.detach.html).

## Question 9 — Gradient accumulation

After y1=2x and the first backward pass, `x.grad = tensor([2.])`.
After y2=3x and the second backward pass, **`x.grad = tensor([5.])`**.
PyTorch accumulates gradients into leaf `.grad` buffers: 2 + 3 = 5.
These are two separate forward graphs, so retaining the first graph is not needed.

Clear the stored gradient between independent updates:

```python
x = torch.tensor([1.0], requires_grad=True)
(x*2).backward()
print(x.grad)  # tensor([2.])
x.grad = None  # alternatively x.grad.zero_()
(x*3).backward()
print(x.grad)  # tensor([3.])
```

For model parameters managed by an optimizer, use
`optimizer.zero_grad(set_to_none=True)` before each independent loss/backward
step. It clears only the parameters included in that optimizer; it does not
reset an unrelated leaf x. Accumulation is useful when intentional, such as
combining microbatches before an optimizer update.

---
Prepared with OpenAI Codex assistance. All example outputs are verified by
executing the corresponding PyTorch code in the submitted notebook.
