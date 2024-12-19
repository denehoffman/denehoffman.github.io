+++
title = "The BFGS Algorithm Family in Rust (Part 2)"
+++

> Since writing the [previous post](@/blog/2024-09-08-the-bfgs-algorithm-family-in-rust-part-1/index.md), I have made several simplifications to the library that will change how some structs and traits are used here. The biggest change is that I've abandoned generic floats in favor of a feature gate:
> ```rust
> #[cfg(not(feature = "f32"))]
> pub type Float = f64;
>  
> #[cfg(feature = "f32")]
> pub type Float = f32;
> ```
> With this code, I can turn the `f32` version of my crate on and off with a feature flag. While generics can be nice, in this instance they were actually causing the crate to be slower (just according to some of my own internal benchmarks, this might not be the case in general) and more difficult to read. Any instances in the last post where there was a generic type `T` representing floating-point values have been replaced with this `Float` type, so `Status<T>` is now just `Status`, for example.

Let's talk about the core BFGS algorithm. The details I'm using here can be found in Nocedal and Wright's book ["Numerical Optimization"](https://doi.org/10.1007/978-0-387-40065-5) in Chapter 6: Quasi-Newton Methods. In the last post, I alluded to this and described gradient descent, but we should really cover what Newton's method is first.

# Newton's Method

The core idea of gradient descent was that we get the gradient at the initial point and walk downhill. We can couple this with a line search to figure out the optimal step size to take downhill, but that's about the best we can do with a single derivative. With two derivatives, however, we can do just a bit better (converge faster). First, let's Taylor expand our function around some point $x$:

$$ f(x + t) \approx f(x) + f'(x)t + \frac{1}{2}f''(x)t^2 $$

We can imagine this as a function of just $t$, keeping $x$ fixed, and try to figure out the value of $t$ that minimizes the function. This can be done by realizing that the derivative of $f(x + t)$ with respect to $t$ should be zero at extrema and the second derivative should be positive at a minimum:

$$ 0 = \frac{\text{d}}{\text{d}t} \left(f(x) + f'(x)t  \frac{1}{2} f''(x)t^2 \right) = f'(x) + f''(x)t $$

or $t = -\frac{f'(x)}{f''(x)}$.

In more than one dimension $n$, we write $f'(x) \to \vec{\nabla}f(\vec{x})$ (a $n$-dimensional vector) and $f''(x) \to \nabla^2 f(\vec{x}) = H$. This second derivative term is an $n \times n$ matrix called the Hessian, and the matrix equivalent of the reciprocal here is the matrix inverse. The Hessian is just a matrix of second derivatives, $H_{ij} = \frac{\partial^2 f(\vec{x})}{\partial \vec{x}_i \partial \vec{x}_j}$, so it is symmetric by definition.

The Hessian is also positive definite, which just means that for all nonzero vectors $\vec{q}$,

$$\vec{q}^{\intercal} H \vec{q} > 0$$

With all of this in mind, recall that the gradient descent update step was

$$ \vec{x}_{k+1} = \vec{x}_k - \alpha_k \vec{\nabla} f(\vec{x}_k) $$

so the equivalent second-derivative (Newton) update step is

$$ \vec{x}_{k+1} = \vec{x}_k - \alpha_k H_k^{-1} \vec{\nabla} f(\vec{x}_k) $$

As far as I know, this is the optimal second-order method for numeric optimization (in theory). In practice, it tends to be horribly inefficient to compute a Hessian matrix and subsequently invert it, and this problem gets worse when you increase the number of free parameters. Particularly, we rarely know the analytic first and second derivatives of our objective function (and it could be argued that knowing those would often make optimization trivial). In most cases, we would have to do these derivatives numerically:

```rust
  fn hessian(&self, x: &[Float], user_data: &mut U) -> Result<DMatrix<Float>, E> {
      let x = DVector::from_column_slice(x);
      let h: DVector<Float> = x
          .iter()
          .map(|&xi| Float::cbrt(Float::EPSILON) * (xi.abs() + 1.0))
          .collect::<Vec<_>>()
          .into();
      let mut res = DMatrix::zeros(x.len(), x.len());
      let mut g_plus = DMatrix::zeros(x.len(), x.len());
      let mut g_minus = DMatrix::zeros(x.len(), x.len());
      for i in 0..x.len() {
          let mut x_plus = x.clone();
          let mut x_minus = x.clone();
          x_plus[i] += h[i];
          x_minus[i] -= h[i];
          g_plus.set_column(i, &self.gradient(x_plus.as_slice(), user_data)?);
          g_minus.set_column(i, &self.gradient(x_minus.as_slice(), user_data)?);
          for j in 0..=i {
              if i == j {
                  res[(i, j)] = (g_plus[(i, j)] - g_minus[(i, j)]) / (2.0 * h[i]);
              } else {
                  res[(i, j)] = ((g_plus[(i, j)] - g_minus[(i, j)]) / (4.0 * h[j]))
                      + ((g_plus[(j, i)] - g_minus[(j, i)]) / (4.0 * h[i]));
                  res[(j, i)] = res[(i, j)];
              }
          }
      }
      Ok(res)
  }
```

I'm not sure if I've written this function as efficiently as possible, but already we can see the issue. To accurately calculate the Hessian, we need to take the gradient in two places for each free parameter (`g_plus` and `g_minus`), a process which requires at least two function calls for each free parameter. That's a total of $4n^2$ function evaluations for $n$ free parameters. If the function takes one second to evaluate and has ten free parameters, the gradient will take twenty seconds, but the Hessian will take over six minutes. Now it might seem silly to speculate on a function which takes a full second to evaluate (at that point, we can expect any optimization problem is going to take a while), but the real issue is scaling. If we add a single free parameter to this problem, it now takes eight minutes. Most of the problems I deal with in my research can be evaluated in less than 200ms, but they contain over 40 free parameters. Do the math, that's 21 minutes for a single evaluation of the Hessian, and we have to do that at every step in the optimization!

Fortunately, there's a better, faster way. Rather than calculating the full Hessian, let's just approximate it! After all, in the limit of very small steps, the Hessian shouldn't be changing that much (for nice functions), so it makes sense that there should be some way to approximate the next Hessian given the current (and the gradients at both points). This is what puts the "quasi" in quasi-Newton methods. We outline some way of approximating the current Hessian given some history of past Hessian approximates and gradients and apply Newton's method for optimization.

Let's refer to this approximate Hessian as $B_k$[^1]. Furthermore, following the derivation in Nocedal and Wright, let's write the second-order Taylor expansion of our function at the point $\vec{x}_k$. I'll use $f_k \equiv f(\vec{x}_k)$ to represent the objective function and $\vec{g}_k \equiv \vec{\nabla}f(\vec{x}_k)$ to represent the gradient.

$$
m_k(\vec{p}) = f_k + \vec{g}_k^{\intercal} \vec{p} + \frac{1}{2} \vec{p}^{\intercal} B_k \vec{p}
$$

This approximation is identical to our function when $\vec{p} = 0$, and furthermore, the gradient is also the same. The only difference is in the second derivatives, which are approximately the same. We can assume that we'll use some line search to find the optimal step length $\alpha_k$ to get to the new point, $\vec{x}_{k+1} = \vec{x}_k + \alpha_k \vec{p}_k$. Let's go to this next point and write out the Taylor expansion for the next step here:

$$
m_{k+1}(\vec{p}) = f_{k+1} + \vec{g}\_{k+1}^{\intercal} \vec{p} + \frac{1}{2} \vec{p}^{\intercal} B_{k+1} \vec{p}
$$

Now imagine we took this step and looked back at the old position and calculated the gradient using this new expansion evaluated at the old position. We would hope to get the same result as when we used our old expansion! We can actually make this a constraint on our matrix $B$. To look back at the old position, we evaluate the function at $p = -\alpha_k\vec{p}_k$, since this is the step we just took but reversed in direction. We can then write the gradient as

$$
\vec{\nabla} m_{k+1}(-\alpha_k\vec{p}\_k) = \vec{g}\_{k+1} - \alpha_k B_{k+1} \vec{p}_k
$$

and with the requirement that $ \vec{\nabla} m_{k+1}(-\alpha_k\vec{p}\_k) = \vec{g}\_k $, we get a new rule for our approximate Hessian:

$$
B_{k+1}\alpha_k\vec{p}\_k = \vec{g}\_{k+1} - \vec{g}\_k
$$

There is some conventional notation to this that we will use throughout the derivation. Rather than using each individual gradient and position, it is helpful to just think about the change in gradient and position, or

$$ \vec{s}_k = \vec{x}\_{k+1} - \vec{x}\_k = \alpha_k\vec{p}\_k $$

and

$$ \vec{y}_k = \vec{g}\_{k+1} - \vec{g}\_k $$

In this notation, the gradient requirement (called the secant equation), is as follows:

$$ B_{k+1}\vec{s}_k = \vec{y}_k $$

Remember when we said the Hessian is positive definite? We would like our Hessian approximate to be positive definite too. Since we don't include any zero-length jumps, $|\vec{s}_k| > 0$, so we can multiply the previous equation by another $\vec{s}_k$ to find:

$$ \vec{s}\_k^{\intercal} B_{k+1} \vec{s}\_k > 0 \implies \vec{s}\_k^{\intercal} \vec{y}_k > 0 $$

This requirement on $\vec{s}\_k$ and $\vec{y}\_k$ is called the curvature condition. As it turns out, there's a fairly simple way to ensure this using the Wolfe condition from my previous article. Recall that during our line search, we required that

$$\begin{align}\vec{p}\_k^{\intercal} \vec{g}\_{k+1} &\geq c_2(\vec{p}\_k^{\intercal} \vec{g}\_k) \\\\ \vec{s}\_k^{\intercal} \vec{g}\_{k+1} &\geq c_2(\vec{s}\_k^{\intercal} \vec{g}\_k) \\\\ \vec{s}\_k^{\intercal} (\vec{y}\_k + \vec{g}\_{k}) &\geq c_2(\vec{s}\_k^{\intercal} \vec{g}\_k) \\\\ \vec{s}\_k^{\intercal} \vec{y}\_k + \vec{s}\_k^{\intercal} \vec{g}\_{k} &\geq c_2(\vec{s}\_k^{\intercal} \vec{g}\_k) \\\\ \vec{s}\_k^{\intercal} \vec{y}\_k &\geq (c_2 - 1)(\vec{s}\_k^{\intercal} \vec{g}\_k) \\\\ \vec{s}\_k^{\intercal} \vec{y}\_k &\geq (c_2 - 1)\alpha\_k \vec{p}\_k \vec{g}\_k \end{align}$$

Since we require $c_2 > 1$ and $\alpha\_k > 0$, and assuming $\vec{p}\_k$ is somewhat aligned with the gradient (or at a minimum, $\vec{p}\_k$ is pointing somewhere downhill), then the curvature condition clearly holds.

Unfortunately, we're only a bit better off than we were when we had no idea what $B$ was. While these conditions limit the space of possible $B$ matrices, the set is still infinite. We need to find more ways to narrow down the space to a unique solution.

One way we could approach this is by assuming that $B$ shouldn't change _too much_ at each update. There are a number of ways of going about this, but the basic idea is that we choose some norm on the change of the matrix between updates with different norms leading to different quasi-Newton algorithms. In particular, minimizing the Frobenius norm for this particular problem will give us the Davidon-Fletcher-Powell (DFP) algorithm. It has a nice update step which we can apply to $B$ that looks like this:

$$B_{k+1} = (I - \rho\_k \vec{y}\_k \vec{s}\_k^{\intercal}) B_k (I - \rho\_k \vec{s}\_k \vec{y}\_k^{\intercal}) + \rho\_k \vec{y}\_k \vec{y}\_k^{\intercal}$$

where $\rho\_k = \frac{1}{\vec{y}\_k^{\intercal} \vec{s}\_k}$. However, we aren't actually done yet, since in the quasi-Newton update step, we really need the matrix $B\_{k+1}^{-1}$. We could go and calculate a matrix inverse at each step, but it turns out we don't need to thanks to the Sherman-Morrison-Woodbury formula, which says that the inverse of a rank-$k$ correction on a matrix is equivalent to a rank-$k$ correction on the inverse, or more precisely,

$$(B + UCV)^{-1} = B^{-1} - B^{-1}U(C^{-1}+VB^{-1}U)^{-1}VB^{-1}$$

where $B$ is $n\times n$, $C$ is $k\times k$, $U$ is $n\times k$, and $V$ is $k\times n$. This is a tough one, but we should take the time to work it out. First we expand the right side of the DFP formula:

$$
B_k - \rho\_k \vec{y}\_k \vec{s}\_k^{\intercal} B_k - \rho\_k B_k \vec{s}\_k \vec{y}\_k^{\intercal} + \rho\_k^2 \vec{y}\_k \vec{s}\_k^{\intercal} B_k \vec{s}\_k \vec{y}\_k^{\intercal} + \rho\_k \vec{y}\_k \vec{y}\_k^{\intercal}
$$

Everything except for the very first $B_k$ should be thought of as those matrices $U$, $V$, and $C$. It is also helpful to know that by a rank-$k$ correction, we mean that the correction is written as $A' = A + \vec{u}\_1\vec{v}\_1^{\intercal} + \vec{u}\_2\vec{v}\_2^{\intercal} + ... + \vec{u}\_k\vec{v}\_k^{\intercal}$. The two terms in the original DFP formula form such a correction of rank-2, since the first terms involve both $\vec{y}\_k$ and $\vec{s}\_k$ scaled by $\rho\_k$ and $B_k$ while the second term only involves $\vec{y}\_k$ scaled by $\rho\_k$, making it linearly independent. We then require $C$ to be a $2\times 2$ matrix, $U$ to be $n\times 2$, and $V$ to be $2\times n$, where $n$ is the dimensionality of the Hessian. It's a bit tricky to see, but we can write this all in the form:[^2]

$$
B_{k+1} = B_k + \begin{bmatrix}\rho\_k\vec{y}\_k & \rho\_k B_k\vec{s}\_k \end{bmatrix} \begin{bmatrix}1 + \rho\_k \vec{s}\_k^{\intercal} B_k \vec{s}\_k & -1 \\\\ -1 & 0 \end{bmatrix} \begin{bmatrix} \vec{y}\_k \\\\ \vec{s}\_k B_k \end{bmatrix}
$$

I'll leave the actual application of the Sherman-Morrison-Woodbury formula as an exercise to the reader, but the conclusion is this rather nice looking update step for the inverse of the (approximate) Hessian:

$$
B_{k+1}^{-1} = B_k^{-1} - \frac{B_k^{-1} \vec{y}\_k \vec{y}\_k^{\intercal} B_k^{-1}}{\vec{y}\_k^{\intercal} B_k^{-1} \vec{y}\_k} + \frac{\vec{s}\_k \vec{s}\_k^{\intercal}}{\vec{y}\_k^{\intercal} \vec{s}\_k}
$$

At this point, we should be pretty happy. Given some starting value for the inverse Hessian, we have a way of determining a new inverse Hessian at our next step given the value of the old one! We could go and write this up, but then this article would need a different title. It just so happens that you can apply this very same process, but swap out $B$ for $B^{-1}$ at every step (with a new secant condition, $ B_{k+1}^{-1}\vec{y}_k = \vec{s}_k $) and skip that last bit entirely, _and it works better in practice_! In other words, the update step,

$$ B_{k+1}^{-1} = (I - \rho\_k \vec{s}\_k \vec{y}\_k^{\intercal})B_{k+1}^{-1} (I - \rho\_k \vec{y}\_k \vec{s}\_k^{\intercal}) + \rho\_k \vec{s}\_k \vec{s}\_k^{\intercal} $$

is also a unique solution to our quasi-Newton problem. This is the BFGS update. Of course, we still have the problem of what values should be assigned to $B_0^{-1}$. The neat part about BFGS is that this update step tends to be self-correcting (DFP is generally worse in this regard), assuming the Wolfe condition is met in the line search (this is why I spent so much time on the line search part of the previous post, it really is that important to make educated leaps in the gradient direction!). What this means is that even if we start with a bad initial guess at $B_0^{-1}$, after some number of steps, we should get to a good approximation regardless. We could of course just start with the numerically calculated Hessian, invert it, and go from there, but in practice, this really isn't worth the effort. Instead, we can just use the identity matrix[^3] for the very first step (just regular gradient descent!) and then correct that identity matrix each time. We will later find that this isn't even the most efficient way of doing things when we look at L-BFGS, but for now we have a formula, let's implement it!

Let's start with a struct to hold our new algorithm:

```rust
#[derive(Clone)]
pub struct BFGS<U, E> {
    x: DVector<Float>,
    g: DVector<Float>,
    h_inv: DMatrix<Float>,
    line_search: Box<dyn LineSearch<U, E>>,
}

impl<U, E> Default for BFGS<U, E> {
    fn default() -> Self {
        Self {
            x: Default::default(),
            g: Default::default(),
            h_inv: Default::default(),
            line_search: Box::<StrongWolfeLineSearch>::default(),
        }
    }
}
```

Next, our update formula:
```rust
impl<U, E> BFGS<U, E> {
    fn update_h_inv(&mut self, step: usize, n: usize, s: &DVector<Float>, y: &DVector<Float>) {
        if step == 0 {
            self.h_inv = self.h_inv.scale((y.dot(s)) / (y.dot(y)));
        }
        let rho = Float::recip(y.dot(s));
        let m_left = DMatrix::identity(n, n) - (y * s.transpose()).scale(rho);
        let m_right = DMatrix::identity(n, n) - (s * y.transpose()).scale(rho);
        let m_add = (s * s.transpose()).scale(rho);
        self.h_inv = (m_left * &self.h_inv * m_right) + m_add;
    }
}
```

I've modified the `Algorithm` trait since making the previous post:

```rust
pub trait Algorithm<U, E>: DynClone {
    fn initialize(
        &mut self,
        func: &dyn Function<U, E>,
        x0: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E>;
    fn step(
        &mut self,
        i_step: usize,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E>;
    fn check_for_termination(
        &mut self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<bool, E>;
    fn postprocessing(
        &mut self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E> {
        Ok(())
    }
}
```

We now just pass the `Minimizer`'s `Status` in as a mutable reference and ask the `Algorithm` to modify it accordingly. The implementation of the BFGS `Algorithm` is simply:

```rust
impl<U, E> Algorithm<U, E> for BFGS<U, E>
where
    U: Clone,
    E: Clone,
{
    fn initialize(
        &mut self,
        func: &dyn Function<U, E>,
        x0: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E> {
        self.f_previous = Float::INFINITY;
        self.h_inv = DMatrix::identity(x0.len(), x0.len());
        self.x = x0;
        self.g = func.gradient(self.x.as_slice(), user_data)?;
        status.inc_n_g_evals();
        status.update_position((
            self.x.as_slice(),
            func.evaluate(self.x.as_slice(), user_data)?,
        ));
        status.inc_n_f_evals();
        Ok(())
    }

    fn step(
        &mut self,
        i_step: usize,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E> {
        let d = -&self.h_inv * &self.g;
        let (valid, alpha, f_kp1, g_kp1) = self.line_search.search(
            &self.x,
            &d,
            Some(self.max_step),
            func,
            bounds,
            user_data,
            status,
        )?;
        if valid {
            let dx = d.scale(alpha);
            let grad_kp1_vec = g_kp1;
            let dg = &grad_kp1_vec - &self.g;
            let n = self.x.len();
            self.update_h_inv(i_step, n, &dx, &dg);
            self.x += dx;
            self.g = grad_kp1_vec;
            status.update_position((self.x.as_slice(), f_kp1));
        } else {
            status.set_converged();
        }
        Ok(())
    }
    // other methods aren't that important for this discussion
}
```

That's pretty much it. Of course, we need to add a few details to tell the algorithm where to stop. Some standard criteria are when the absolute change in the function value or the norm of the gradient goes below some tolerance, and you can see those termination conditions implemented in the full code [here](https://github.com/denehoffman/ganesh/blob/c5011b52668ba6ee73444bf5a754e51f20142557/src/algorithms/bfgs.rs).

> I have to apologize a bit for the amount of changes in the code that have taken place since the last article. I'm developing this in conjunction with a library I'm currently using for amplitude analysis of particle physics data, and being relatively new to Rust, I often find improvements by realizing that I've backed myself into an implementation corner. I hope this article serves as a nice introduction to the BFGS algorithm and how one _might_ implement it in Rust, but it is likely not the best way to implement this or the other algorithms I've discussed. If any readers see places where my code could be improved, I would welcome suggestions [via the GitHub repo](https://github.com/denehoffman/ganesh/issues).

With that, I'll end this portion of the series. In the next post, I'll discuss the L-BFGS algorithm, the "limited memory" version of BFGS. The surprising thing we will see is that this method tends to work better than BFGS in most practical settings, and its close, bounded cousin, L-BFGS-B also tends to outperform standard BFGS updates. This is mostly because these methods not only reduce the memory required to perform BFGS updates (for large parameter spaces), they tend to involve fewer operations, which makes them more efficient to calculate. The core idea will be, rather than update and store the entire inverse Hessian, just store the changes and update an identity matrix, and while we're at it, just calculate the update step directly from the stored changes to the gradient!


[^1]: This notation is somewhat standard, but if you spend any time reading old papers on this, you'll find that they tend to use $B$, $H$, and their inverses rather interchangeably. For the sake of clarity, $H$ here will always refer to the true Hessian and $B$ to some approximated Hessian.

[^2]: Credit due to [A.Γ.](https://math.stackexchange.com/users/253273/a-%ce%93) in [this post](https://math.stackexchange.com/a/2785578/127253) on the Mathematics StackExchange

[^3]: While this would work, it's recommended to actually use $\frac{\vec{y}\_k^{\intercal}\vec{s}\_k}{\vec{y}\_k^{\intercal}\vec{y}\_k}$ for the very first step, after the step has been made but before the first update is performed, and this is reflected in the code.
