+++
title = "The BFGS Algorithm Family in Rust (Part 3)"
description = "The L-BFGS-B implementation"
[taxonomies]
tags = ["rust", "algorithms", "programming"]
+++

In my previous posts ([here](@/blog/2024-09-08-the-bfgs-algorithm-family-in-rust-part-1/index.md) and [here](@/blog/2024-12-15-the-bfgs-algorithm-family-in-rust-part-2/index.md)), I discussed line searches and the original BFGS algorithm. The next improvement on this algorithm is a limited memory version, called L-BFGS, but rather than explicitly constructing it, we will instead skip to the limited memory version with parameter bounds, L-BFGS-B. However, we should start by understanding why there even is an L-BFGS algorithm at all.

In the BFGS algorithm, we start with a guess for the inverse Hessian $B_0^{-1}$ and then improve this guess at each step by defining

$$ s_k = x_{k+1} - x_k,$$

$$ y_k = g_{k+1} - g_k,$$

and

$$ \rho = \frac{1}{\vec{y}_k^{\intercal} \vec{s}_k}, $$

and updating according to the formula

$$ B_{k+1}^{-1} = \left(I - \rho_k \vec{s}\_k \vec{y}\_k^{\intercal} \right) B_k^{-1} \left(I - \rho_k \vec{y}\_k \vec{s}\_k^{\intercal} \right) + \rho_k \vec{s}\_k \vec{s}\_k^{\intercal}. $$

This doesn't seem like it uses too much memory. At worst, the Hessian must be symmetric, so for $n$ parameters, we need $n(n+1)/2$ stored values, and if each is a 64-bit float, that's just over $40$ kilobytes for $100$ free parameters. However, we must also consider the storage limitations of the 1990s, where typical computers had maybe $128$ megabytes of RAM at a maximum. Some part of that RAM had to be used to store any data used in the minimization, part of it had to store the program itself, and part of it was allocated to the other operations of the computer. We live in an era where most laptops have a minimum of $4$ gigabytes of RAM, and it would take a $16000$-parameter model just to use $1$ gigabyte of that. There are other considerations, for instance, you actually need to allocate twice as much memory since you need some place to store the next iteration. It turns out, there is a way to greatly reduce the amount of memory needed, and this is what Byrd, Nocedal, and Schnabel[^1] did back in 1989.

## The L-BFGS Algorithm

Their method is centered around the following idea: if you were to store every $\vec{s}$ and $\vec{y}$ and recompute the inverse Hessian from scratch at each step, only the most recent updates to those vectors would be very useful. If we're 100 steps into the algorithm, we could imagine starting the algorithm at step 90 with the identity matrix initial guess, and eventually we will converge upon the correct Hessian again. In fact, they found that this convergence tends to happen very quickly, typically in less than 10 steps or so. Therefore, instead of storing the entire inverse Hessian, we can instead store some finite number of $\vec{s}$ and $\vec{y}$ entries, discarding the oldest ones, and calculate the inverse Hessian from scratch each time. If we store $m$ entries, recall that each vector contains $n$ values, so we now need to only store $2n\times m$ values rather than $n(n+1)/2$. If we set $m=10$ and again use 64-bit floats, a $100$-parameter model would only need $16$ kilobytes rather than $40$. The real gains are obvious in the case of extremely large models. For instance, the $16000$-parameter model would now only take $2.6$ megabytes rather than $1$ gigabyte. What do we lose here? Well, we typically have to do a few more operations, but we get improvements in cache locality, since we no longer have to transfer a large matrix to the local cache at each step. We also save on allocations, since we don't actually need to construct the inverse Hessian at all, and we can just construct the update step directly instead.

As mentioned, we could go through the entire algorithm separately, but it's so closely related to L-BFGS-B that we will just implement that directly instead. This article will follow the work of Byrd, Lu, Nocedal, and Zhu[^2], particularly their notation, except for my usage of $B^{-1}$ as the inverse Hessian approximate rather than their use of $H$ for that matrix.

First, we define our registers of $\vec{s}$ and $\vec{y}$ as $n\times m$ matrices:

$$ Y_k = [\vec{y}\_{k-m},\ldots,\vec{y}\_{k-1}] $$

and

$$ S_k = [\vec{s}\_{k-m},\ldots,\vec{s}\_{k-1}] $$

Just to clarify, these are matrices where each row corresponds to a particular parameter and each column corresponds to an algorithm step, where the leftmost column is the oldest step and the rightmost column is the most recent. If we let $\theta$ be a positive scaling parameter, then as long as each pair of $\vec{s}$ and $\vec{y}$ satisfy $\vec{s}^{\intercal} \vec{y} > 0 $, we can write the Hessian approximate as

$$ B_k = \theta I - W_k M_k W_k^{\intercal} $$

where

$$ W_k = \begin{bmatrix} Y_k & \theta S_k \end{bmatrix} $$

is a block matrix of dimension $n\times 2m$ and

$$ M_k = \begin{bmatrix} -D_k & L_k^{\intercal} \\\\ L_k & \theta S_k^{\intercal} S_k \end{bmatrix}^{-1} $$

is another block matrix of dimension $2m\times 2m$, where

$$ (L_k)\_{i,j} = \begin{cases} (s_{k-m-1+i})^{\intercal} (y_{k-m-1+j}) & \text{if } i > j \\\\ 0 & \text{otherwise}\end{cases} $$

and

$$ D_k = \text{diag}\left[ s_{k-m}^{\intercal} y_{k-m},\ldots,s_{k-1}^{\intercal} y_{k-1} \right] $$

Woah, slow down! What's going on here? This is the notation used in Byrd et al., and it's pretty confusing. Let's ignore the intricacies of $M_k$ and just focus on the blocks. $L_k$ is upper triangular with zeros on the diagonal, that's all the case-statement means. All of those subscripts look confusing, but if you recall, $k-m$ is just shorthand for the oldest index stored and $k-1$ is the newest index, so $L_k$ is really just the upper triangular block of $S_k^{\intercal} Y_k$! Similarly, $D_k$ is just the diagonal of that matrix, so to get all of these parts, we just need to perform the matrix product and pull apart the relevant blocks. The fact that $M_k$ is the inverse of this matrix seems a bit tricky, but recall that $M_k$ has dimension $2m \times 2m$, and if we pick $m$ to be small, the cost to compute the inverse is actually pretty low (we can actually make $m$ arbitrarily small (but $ >1 $), and all we will lose is the speed of convergence).

I'm not going to explain exactly why this set of matrices works, but hopefully you can see that we end up doing something similar to the original BFGS update step, where the Hessian is approximated by some correction to the identity.

While I won't show the entire Rust code for this, I'll give you the general idea:

```rust
// First, W_k
let w_mat = DMatrix::zeros(n, 2 * m);
// We can use view_mut to update a submatrix
let mut y_view = w_mat.view_mut((0, 0), (n, m));
y_view += &y_mat;
let mut theta_s_view = w_mat.view_mut((0, m), (n, m));
theta_s_view += s_mat.scale(theta);

// Next, we calculate M_k
// S_k^T Y_k:
let s_tr_y = s_mat.transpose() * &y_mat;
// D_k:
let d_vec = s_tr_y.diagonal();
// L_k^T:
let mut l_tr_mat = s_tr_y.lower_triangle();
l_tr_mat.set_diagonal(&DVector::from_element(m, 0.0)); // clear the diagonal
// theta * S_k^T S_k:
let theta_s_tr_s = (s_mat.transpose() * &s_mat).scale(theta);
// M_k^{-1}:
let mut m_mat_inv = DMatrix::zeros(2 * m, 2 * m);
let mut d_view = m_mat_inv.view_mut((0, 0), (m, m));
d_view.set_diagonal(&(-&d_vec));
let mut l_tr_view = m_mat_inv.view_mut((m, 0), (m, m));
l_tr_view += &l_tr_mat;
let mut l_view = m_mat_inv.view_mut((0, m), (m, m));
l_view += l_tr_mat.transpose();
let mut theta_s_tr_s_view = m_mat_inv.view_mut((m, m), (m, m));
theta_s_tr_s_view += theta_s_tr_s;
let m_mat = m_mat_inv.try_inverse().expect("inv(M) failed!");
```

In the implementation, we also store `w_mat` and `m_mat`, since `m_mat` is just $2m \times 2m$ and `w_mat` is $n \times 2m$, so they're both relatively small for small $m$. Another thing to note is that the authors suggest discarding correction pairs $\vec{s}_k$ and $\vec{y}_k$ if they do not satisfy the curvature condition

$$ \vec{s}\_k^{\intercal} \vec{y}\_k > \varepsilon ||\vec{y}_k||^2 $$

for some small $\varepsilon > 0$. We technically have a way to compute the inverse Hessian approximate, and we could just stop there and redo the BFGS update step, giving us the L-BFGS algorithm. Let's instead move on to the bounded version, and we'll show how we can avoid constructing the inverse Hessian approximate entirely by just directly updating the descent step.

## Adding Boundaries

There are a couple of ways one might add bounded parameter support to a minimization algorithm. Algorithms like MINUIT and LMFIT use a nonlinear transform which stretches the distance to a boundary infinitely as one approaches the bound. One advantage of this is that it can generally be adapted to any minimization method, but the major disadvantage is that it turns any linear problem into a nonlinear one. This also makes it difficult to properly reconstruct the Hessian later for uncertainty analysis, and it can be very slow when the minimum is close to or on a boundary.

Another way is to pin parameters at their boundary value when an update step would try to cross that boundary. This is kind of what we'll end up doing, but it's not quite so straightforward when we want to be memory/computationally efficient.

### Computing the Generalized Cauchy Point

The first step is similar to any gradient-descent algorithm. We need to compute a step direction, and originally we'd just go in the opposite direction of the gradient. However, the gradient might point us towards a boundary, or we might already be on a boundary. Remember, we are modeling the objective function locally as a quadratic and jumping as close to the minimum as we can. Byrd et al. start this process by computing "breakpoints":

$$ t_i = \begin{cases}
(x_i - u_i) / g_i & \text{if } g_i < 0\\\\
(x_i - l_i) / g_i & \text{if } g_i > 0\\\\
\infty & \text{if } g_i = 0
\end{cases} $$

I would forgive the reader if they can't figure out why we are doing this, I also had no idea what was going on here when I first read it. We calculate the gradient $\vec{g}$, and we want to move in the opposite direction. For each component of $\vec{g}$, $g_i < 0$ would imply that we are moving in the positive direction, so we might have to worry about an upper bound. $(x_i - u_i)$ is the distance to that upper bound (but note that the order makes it a negative value as long as we haven't crossed this bound!), and we divide this by the length of the step (which is also negative according to the condition, so this quantity is strictly positive). Similarly, if the step is in the other direction, we get another strictly positive value with the same kind of proportionality. Assuming no step ever puts us over a boundary, can get a minimum value of zero if we are already on the boundary, otherwise we will get larger positive values the further away from the boundary we are (scaled by the size of the step in that direction).

Next, we're going to sort the indices by these values, but first let's additionally calculate a vector $\vec{d}$ which follows the formula,

$$ d_i = \begin{cases}
0 & \text{if } t_i = 0\\\\
-g_i & \text{otherwise}
\end{cases} $$

Recall that $t_i = 0$ means $x_i$ is already at a boundary, so this $\vec{d}$ is just the descent vector (negative gradient) in all directions that aren't at a boundary. In rust, this looks something like,
```rust
let (t, mut d): (DVector<Float>, DVector<Float>) = (0..g.len()).map(|i| {
  let t_i = if g[i] < 0.0 {
    (x[i] - u[i]) / g[i]
  } else if g[i] > 0.0 {
    (x[i] - l[i]) / g[i]
  } else {
    Float::INFINITY
  };
  let d_i = if t_i < Float::EPSILON { 0.0 } else { -g[i] };
  (t_i, d_i)
})
```
We use $t_i < \epsilon$ here rather than $t_i = 0$ to avoid issues with comparing floating point values. The point is, if we are close enough to a boundary, we probably shouldn't try to move in that direction, even if we are technically not right on it. `d` is mutable for reasons which will become clear later.

We next need to establish a list of "free" indices $\mathcal{F} = \left\\{i \mid t_i > 0\right\\}$. These are the indices of all parameter which are not at a boundary. This next part is a bit tricky, and I'm not going to derive it because this article would be much too long (and you can see the derivation in [^2] anyway), but I'll give you the general idea. We are approximating the function as a quadratic

$$ m_k(x) = f(x_k) + g_k^{\intercal}(x - x_k) + \frac{1}{2}(x - x_k)^{\intercal}B_k(x - x_k) $$

but instead of minimizing it for all $x$, we instead want to minimize it just for feasible values of $x$ (values within the boundaries). We therefore construct a piecewise function $\vec{x}(t) = P(\vec{x}_k - t \vec{g}_k, \vec{l}, \vec{u})$ where $P(\vec{a}, \vec{l}, \vec{u})$ just clips $\vec{a}$ inside a box defined by $\vec{l}$ and $\vec{u}$. We then want to find the first minimizer of $q_k(t) = m_k(\vec{x}(t))$, which we call the generalized Cauchy point $\vec{x}^c$. This can be thought of as the first feasible point along the search direction (accounting for boundaries). After this, we still need to minimize the objective function along this direction. Imagine that we are walking along the gradient and we periodically hit a boundary in some dimension. Each time we hit a boundary, we pin that parameter to the boundary and keep walking in the same direction for all other parameters. This creates a piecewise-linear path for which the generalized Cauchy point is the first valid point we compute. Now it might make more sense why we defined each $t_i$ as above. We will eventually sort these $t_i$ to determine the order in which to drop free indices, and travel along this piecewise path to find the minimum! Each $t_i$ represents the distance along the piecewise path in units of the gradient which must be traveled to hit a boundary.

We initialize some vector $\vec{p} = W^{\intercal} \vec{d}$ which is a vector of dimension $2m$ which holds $Y_k \vec{d}$ in the first $m$ indices and $\theta S_k \vec{d}$ in the next $m$ indices. We also keep track of a vector $c$ initialized to zero with the same dimensions as $\vec{p}$. We can calculate the derivative of the objective function at the current point to be $f' = -\vec{d}^{\intercal}\vec{d}$ and the second derivative as $f'' = -\theta f' - \vec{p}^{\intercal} M \vec{p}$ (this comes from a simplification of $f'' = \theta \vec{d}^{\intercal}\vec{d} - \vec{d}^{\intercal} WMW^{\intercal} \vec{d}$). We can then find the distance to the minimum along this direction, $\Delta t_{\text{min}} = - \frac{f'}{f''}$.

This gives us the distance along the first piecewise-linear segment which minimizes the quadratic approximation, so now we need to check the subsequent segments. We do this by sorting $\mathcal{F}$ by the values of $t_i$ (the smallest $t_i$ corresponds to the parameter closest to a boundary in units of the gradient). We remove the smallest one from the set of free indices and call it $b$. We define $\Delta t = t_b - 0$ for this first segment, and this corresponds to the distance we can travel in $t$ till we get to the next boundary (not the one we just explored). As long as the distance to the minimum of the current segment is less than the distance to the next segment ($\Delta t$) and we still have unexplored segments, we keep going.

The update step for each new segment becomes a bit more complicated, since we are now displaced from the start of the algorithm at some point along the piecewise-linear path. We define a point $z_b$ as the distance along parameter $b$ from the starting point to the boundary. We update $\vec{c} \to \vec{c} + \Delta t \vec{p}$ and update the function derivatives according to

$$ f' = f' + \Delta t f'' + g_b^2 + \theta g_b z_b - g_b \vec{(w_b)}^{\intercal}M\vec{c} $$

and

$$ f'' = f'' - \theta g_b^2 - 2g_b \vec{(w_b)}^{\intercal}M\vec{p} - g_b^2 \vec{(w_b)}^{\intercal}M \vec{(w_b)} $$

Here we define $\vec{(w_b)}$ as the $b$th row of the $W_k$ matrix, and we use the previous values of $f'$ and $f''$ to calculate their updates. Then we update $\vec{p} \to \vec{p} + g_b \vec{(w_b)}$ and we set the $b$th element of $\vec{d}$ to zero (this is why `d` is mutable in the code above) since we are essentially saying that this parameter is no longer part of the set of free parameters. We then repeat the same process as before to find the minimum along this segment, we remove the next free index, and explore the following segment in that direction as long as it is smaller than the distance we just traveled along the current segment. Notice that if a parameter is unbounded or very far away from a bound, this condition cannot be met, so the algorithm will terminate.

After we have explored all viable segments, we calculate the generalized Cauchy point as

$$ x_i^c = x_i + t_{\text{tot}} d_i $$

for all $i$ with $t_i \geq t$ (all remaining segments) where $t_{\text{tot}}$ is the total distance traveled along the path. We do one final update to $\vec{c}$ before continuing to the next step, subspace minimization.

The full code looks something like this:

```rust
fn get_generalized_cauchy_point(
    x: &DVector<Float>,
    g: &DVector<Float>,
    l: &DVector<Float>,
    u: &DVector<Float>,
    w_mat: &DMatrix<Float>,
    m_mat: &DMatrix<Float>,
    theta: Float,
) -> (DVector<Float>, DVector<Float>, Vec<usize>) {
    // Equations 4.1 and 4.2
    let (t, mut d): (DVector<Float>, DVector<Float>) = (0..g.len())
        .map(|i| {
            let ti = if g[i] < 0.0 {
                (x[i] - u[i]) / g[i]
            } else if g[i] > 0.0 {
                (x[i] - l[i]) / g[i]
            } else {
                Float::INFINITY
            };
            let di = if ti < Float::EPSILON { 0.0 } else { -g[i] };
            (ti, di)
        })
        .unzip();
    let mut x_cp = x.clone();
    let mut free_indices: Vec<usize> = (0..t.len()).filter(|&i| t[i] > 0.0).collect();
    free_indices.sort_by(|&a, &b| t[a].total_cmp(&t[b]));
    let free_indices = VecDeque::from(free_indices);
    let mut t_old = 0.0;
    let mut i_free = 0;
    let mut b = free_indices[0];
    let mut t_b = t[b];
    let mut dt_b = t_b - t_old;

    let mut p = w_mat.transpose() * &d;
    let mut c = DVector::zeros(p.len());
    let mut df = -d.dot(&d);
    let mut ddf = (-theta).mul_add(df, -p.dot(&(&m_mat * &p)));
    let mut dt_min = -df / ddf;

    while dt_min >= dt_b && i_free < free_indices.len() {
        // b is the index of the smallest positive nonzero element of t, so d_b is never zero!
        x_cp[b] = if d[b] > 0.0 { u[b] } else { l[b] };
        let z_b = x_cp[b] - x[b];
        c += p.scale(dt_b);
        let g_b = g[b];
        let w_b_tr = w_mat.row(b);
        df += dt_b.mul_add(
            ddf,
            g_b * (theta.mul_add(z_b, g_b) - w_b_tr.transpose().dot(&(&m_mat * &c))),
        );
        ddf -= g_b
            * theta.mul_add(
                g_b,
                (-2.0 as Float).mul_add(
                    w_b_tr.transpose().dot(&(&m_mat * &p)),
                    -(g_b * w_b_tr.transpose().dot(&(&m_mat * w_b_tr.transpose()))),
                ),
            );
        p += w_b_tr.transpose().scale(g_b);
        d[b] = 0.0;
        dt_min = -df / ddf;
        t_old = t_b;
        i_free += 1;
        if i_free < free_indices.len() {
            b = free_indices[i_free];
            t_b = t[b];
            dt_b = t_b - t_old;
        } else {
            t_b = Float::INFINITY;
        }
    }
    dt_min = Float::max(dt_min, 0.0);
    t_old += dt_min;
    for i in 0..x.len() {
        if t[i] >= t_b {
            x_cp[i] += t_old * d[i];
        }
    }
    let free_indices = (0..free_indices.len())
        .filter(|&i| x_cp[i] < u[i] && x_cp[i] > l[i])
        .collect();
    c += p.scale(dt_min);
    (x_cp, c, free_indices)
}
```

### Subspace Minimization with the Primal Direct Method

The authors of [^1] list three different ways to actually minimize the function after finding the generalized Cauchy point, the direct primal method, a method using conjugate gradients, and a dual method. We will only go over the direct primal method, since it's the most common implementation in practice and the method I use in my own implementation.

The details of this algorithm are quite daunting, but the general idea is that we can build a "reduced" Hessian of the quadratic approximate $m_k(x)$ which only acts on parameters which are not already at a boundary and then carry out a minimization over the subspace of free parameters. To do this, we start by constructing a matrix $Z_k$ which has dimensions $n\times n_\text{free}$ where

$$ (Z_k)_{i,j} = \begin{cases} 1 & \text{if } i = j \text{ and } j\in\mathcal{F} \\\\ 0 & \text{otherwise} \end{cases} $$

We can then imagine operations $\hat{B}_k = Z_k^{\intercal} B_k Z_k$ which take the Hessian as a $n\times n$ matrix to a reduced Hessian of dimension $n_\text{free}\times n_\text{free}$. We can compute a reduced gradient at the generalized Cauchy point as

$$ \hat{r}^c = Z_k^{\intercal} (g_k + \theta(x^c - x_k) - W_k M_k \vec{c}) $$

We then want to minimize

$$ \hat{m}_k(\hat{d}) \equiv \hat{d}^{\intercal} \hat{r}^c + \frac{1}{2}\hat{d}^{\intercal} \hat{B}_k \hat{d}$$

subject to $l_i - x_i^c \leq \hat{d}_i \leq u_i - x_i^c$ for $i\in\mathcal{F}$.

It can be shown (though I will certainly not show the details) that

$$ \hat{d}^u = -\left(\frac{1}{\theta}\hat{r}^c + \frac{1}{\theta^2}Z^{\intercal} W \left( I - \frac{1}{\theta} MW^{\intercal}ZZ^{\intercal}W \right)^{-1} MW^{\intercal}Z\hat{r}^c\right) $$

is the direction in the subspace which minimizes $\hat{m}_k$. Note the leading minus sign! This cannot be found in equation 5.11 or step 6 of the direct primal method algorithm in [^2], but I believe this is a typo. This minus sign is correct, and I've confirmed this by looking at several implementations of this algorithm in several different languages. It is also correct within the paper itself, as can be seen in equation 5.7, which states $\hat{d}^u = -\hat{B}_k^{-1}\hat{r}^c$, combined with equation 5.10 which gives the definition for $\hat{B}^{-1}$. This was absolutely painful to debug, and I hope that if anyone feels the need to go through this paper again, they'll see this and not have the same issues I did when I first coded this up.

The code for this step looks like:

```rust
fn direct_primal_min(
    x: &DVector<Float>,
    g: &DVector<Float>,
    l: &DVector<Float>,
    u: &DVector<Float>,
    w_mat: &DMatrix<Float>,
    m_mat: &DMatrix<Float>,
    theta: Float,
    x_cp: &DVector<Float>,
    c: &DVector<Float>,
    free_indices: &[usize],
) -> DVector<Float> {
    let z_mat = DMatrix::from_fn(x.len(), free_indices.len(), |i, j| {
        if i == free_indices[j] {
            1.0
        } else {
            0.0
        }
    });
    let r_hat_c = z_mat.transpose()
        * (&g + (x_cp - &x).scale(theta) - &w_mat * &m_mat * c);
    let w_tr_z_mat = w_mat.transpose() * &z_mat;
    let n_mat = DMatrix::identity(m_mat.shape().0, m_mat.shape().1)
        - (&m_mat * (&w_tr_z_mat * w_tr_z_mat.transpose())).unscale(theta);
    let n_mat_inv = n_mat
        .try_inverse()
        .expect("Error: Something has gone horribly wrong, inversion of N^{-1} failed!");
    let v = n_mat_inv * &m_mat * &w_mat.transpose() * &z_mat * &r_hat_c;
    let d_hat_u =
        -(r_hat_c + (w_tr_z_mat.transpose() * v).unscale(theta)).unscale(theta);
    // The minus sign here is missing in equation 5.11, this is a typo!
    let mut alpha_star = 1.0;
    for i in 0..free_indices.len() {
        let i_free = free_indices[i];
        alpha_star = if d_hat_u[i] > 0.0 {
            Float::min(alpha_star, (u[i_free] - x_cp[i_free]) / d_hat_u[i])
        } else if d_hat_u[i] < 0.0 {
            Float::min(alpha_star, (l[i_free] - x_cp[i_free]) / d_hat_u[i])
        } else {
            alpha_star
        }
    }
    let mut x_bar = x_cp.clone();
    let d_hat_star = d_hat_u.scale(alpha_star);
    let z_d_hat_star = &z_mat * d_hat_star;
    for i in free_indices {
        x_bar[*i] += z_d_hat_star[*i]
    }
    x_bar
}
```

The rest of the algorithm is fairly similar to the BFGS algorithm. We get a search direction, which is determined either by `x_bar - x` (or `x_cp - x` if there are no remaining free indices, which is why we don't just return the search direction immediately), then we do a line search along this direction, then we move and check if the termination conditions are met. We obtain a new update for $\vec{s}_k$ and $\vec{y}_k$ and get rid of the oldest one if the new one meets the curvature condition, and repeat!

The link to the full implementation is [here](https://github.com/denehoffman/ganesh/blob/2640f0cb87560df3640657d73386f1d79c2358a6/src/algorithms/lbfgsb.rs). Note that this is a permalink to the most recent commit at time of writing, but I plan to make a rather large update to the API which will slightly alter some of the structure of the algorithm.

There are likely several places where I could optimize the code, but I haven't really gotten to that yet. As far as I know, this is the first and possibly only pure Rust implementation of the L-BFGS-B algorithm (if someone else has a better claim, I'll recant), and there is probably some room for improvement. If you find something that could be optimized, let me know, make a PR, send me an email!

This is the last post in my first series of blog posts on this site. As you might be able to tell, I'm still getting used to writing regularly, and I've been a bit preoccupied with my PhD thesis lately, but I hope to have more interesting posts in the future (which will hopefully be written with a bit more forethought than these have been). I hope anyone reading this has enjoyed it, and I hope it clears up some of the general ideas of each algorithm. If anyone is interested in a more in-depth review, check out the cited papers here or any of the citations in the previous posts, as they are full of interesting notes and insights.

[^1]: [R. H. Byrd, J. Nocedal, and R. B. Schnabel, “Representations of quasi-Newton matrices and their use in limited memory methods,” Mathematical Programming, vol. 63, no. 1–3, pp. 129–156, Jan. 1994, doi: 10.1007/bf01582063.
](https://doi.org/10.1007/bf01582063)
[^2]: [R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu, “A Limited Memory Algorithm for Bound Constrained Optimization,” SIAM J. Sci. Comput., vol. 16, no. 5, pp. 1190–1208, Sep. 1995, doi: 10.1137/0916069.](https://doi.org/10.1o137/0916069)
