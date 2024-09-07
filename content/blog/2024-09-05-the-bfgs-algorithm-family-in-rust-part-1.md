+++
title = "The BFGS Algorithm Family in Rust (Part 1)"
+++

The BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm and its derivatives were (and for the most part still are) the gold standard methods for quasi-Newton optimization. In this post, I want to give a brief overview of the main idea, the limited-memory adaptation (L-BFGS), and the bounded version (L-BFGS-B) and how I implemented them in a Rust crate I'm developing called [`ganesh`](https://github.com/denehoffman/ganesh). The full algorithm can be seen there, and I will mainly be focusing on the main methodology, since the actual literature on it is rather old and difficult to parse (and even has a few typos!).

That being said, I wouldn't have done it without the following articles/projects which have guided my understanding:

- ["Numerical Optimization"](https://doi.org/10.1007/978-0-387-40065-5) by Jorge Nocedal and Stephen J. Wright
- ["Numerical Optimization: Understanding L-BFGS"](https://aria42.com/blog/2014/12/understanding-lbfgs) by [Aria Haghighi](https://aria42.com/)
- [L-BFGS-B in pure Python](https://github.com/avieira/python_lbfgsb/) by [@avieira (Alex Vieira?)](https://github.com/avieira)
- [L-BFGS-B in pure MATLAB](https://github.com/bgranzow/L-BFGS-B) by [Brian Granzow](https://github.com/bgranzow)
- ["A Limited Memory Algorithm for Bound Constrained Optimization"](https://doi.org/10.1137/0916069) by Richard H. Byrd, Peihuang Lu, Jorge Nocedal, and Ciyou Zhu

# Quasi-Newton?

Typically, in numeric optimization problems, we want to define some stepping process that will cause us to approach a minimum or maximum as quickly as possible. For our purposes, we'll just assume we only care about minimizing, and we'll also assume that there *is* a single global minimum to the function in question, or at least that we don't care about falling into a deep local minimum. The simplest non-trivial way to do this is by gradient descent:
$$
\vec{x}_{k+1} = \vec{x}_k - \alpha_k \vec{\nabla} f(\vec{x}_k)
$$
where $f: \mathbb{R}^n \to \mathbb{R}$ is the objective function we are trying to minimize and $\alpha_n$ is some positive step length (also called the learning rate). The minus sign here is why we call it gradient descent; we are always moving opposite the gradient, which always points uphill. For simplicity, we'll also refer to the gradient as a function $\vec{g}(\vec{x}) \equiv \vec{\nabla}f(\vec{x})$ Now if you just throw in some very small value for $\alpha$ and cross your fingers, you might eventually end up at the function's minimum, but it certainly won't be the most efficient way to get there. If your $\alpha$ is too big, you could end up overshooting the minimum and bouncing back and forth around it endlessly.

There are several ways we can optimize the choice of step length, but we will be implementing an algorithm that attempts to satisfy the Strong Wolfe conditions. These are conditions for accepting a step length given some step direction $\vec{p}_k$ (we'll see later why we need to generalize this, but for now you can always just imagine $\vec{p}_k = -g(\vec{x}_k)$).

The first of these conditions is also called the Armijo rule:
$$
f(\vec{x}_k + \alpha_k \vec{p}_k) \leq f(\vec{x}_k) + c_1 \alpha_k \left(\vec{p}_k \cdot \vec{g}(\vec{x}_k)\right)
$$
for some value $0 < c_1 < 1$. The usual choice of $c_1$ is $10^{-4}$, which I believe just comes from some experimentation on standard test functions. This method is also called the sufficient decrease condition, and we can see why. The left-hand side is the function value at the new location, which we hope is at least smaller than the previous location (otherwise we are ascending!). However, for it to be sufficiently smaller, the difference must exceed the final term in the equation, which is usually going to be negative due to that dot product.

The second condition, dubbed the curvature condition, requires that the gradient of the function decrease sufficiently. This is usually harder to accomplish, so when we implement this in Rust, we will make it optional but desired.
$$
-\left(\vec{p}_k \cdot \vec{g}(\vec{x}_k + \alpha_k \vec{p}_k)\right) \leq -c_2 \left(\vec{p}_k \cdot \vec{g}(\vec{x}_k)\right)
$$
This condition adds another hyperparameter, $0 < c_1 < c_2 < 1$ where $c_2 = 0.9$ in most applications. However, if we really want to find the best point, we should try to satisfy the **strong** version of the curvature condition:
$$
\left|\vec{p}_k \cdot \vec{g}(\vec{x}_k + \alpha_k \vec{p}_k)\right| \leq c_2 \left|\vec{p}_k \cdot \vec{g}(\vec{x}_k)\right|
$$

# Rust implementation

## Function trait

Let's start by defining a trait which will evaluate our function $f$ and its gradient $\vec{g}$. I want the input parameter values $\vec{x}$ to be generic so that both `f64` and `f32` types can be used, as well as any other struct with the right trait implementations. I also want these functions to return `Result`s with a generic error type `E` so that users can handle any errors their own functions create. Finally, we should consider adding optional arguments to these functions. Again, we will turn to generics, but allow users to pass a `&mut U` called `user_data`. This term is mutable because it will give us the most flexibility later on. The trait for a function might look something like this:

```rust
use num::{traits::NumAssign, Float, FromPrimitive};

pub trait Function<T, U, E>
where
    T: Float + FromPrimitive + Debug + NumAssign,
{
    fn evaluate(&self, x: &[T], user_data: &mut U) -> Result<T, E>;
 
    fn gradient(&self, x: &[T], user_data: &mut U) -> Result<Vec<T>, E> {
        let n = x.len();
        let mut grad = vec![T::zero(); n];
        let h: Vec<T> = x
            .iter()
            .map(|&xi| T::cbrt(T::epsilon()) * (if xi == T::zero() { T::one() } else { xi }))
            .collect();
        for i in 0..n {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();
            x_plus[i] += h[i];
            x_minus[i] -= h[i];
            let f_plus = self.evaluate(&x_plus, user_data)?;
            let f_minus = self.evaluate(&x_minus, user_data)?;
            grad[i] = (f_plus - f_minus) / (convert!(2.0, T) * h[i]);
        }
        Ok(grad)
    }
}
```

I also use a little macro to convert raw numeric fields to our generic type `T`, if possible:
```rust
#[macro_export]
macro_rules! convert {
    ($value:expr, $type:ty) => {{
        #[allow(clippy::unwrap_used)]
        <$type as num::NumCast>::from($value).unwrap()
    }};
}
```

Let's walk through the anatomy of the trait above. I think the `evaluate` method is pretty self-explanatory (given that it's an empty template), but the gradient method is a bit more complex. First of all, I'm implementing a central finite-difference here:
$$
\frac{\partial f(\vec{x})}{\partial x_i} = \frac{f(\vec{x} + h_i \hat{e}_i) - f(\vec{x} - h_i \hat{e}_i)}{2h_i}
$$
The tricky detail is choosing a value for $h_i$. In practice, machine epsilon is too small! What we actually should use is $h_i = \sqrt[3]{\varepsilon} x_i $ when $x_i \neq 0$ and $h_i = \sqrt[3]{\varepsilon}$ in the event that $x_i = 0$.

## Algorithm Trait

Next, since we want to implement three algorithms with very similar features, it might make sense to create a generic trait that can be used by some executor that will wrap all of these methods into a nice API. All of these algorithms will will need to know the following:

1. The objective `Function`
2. The starting point $x_0$
3. Any bounds on the free parameters (we will ignore bounds for the BFGS and L-BFGS methods here, although an experimental change of variables is implemented in the final crate)
4. The `user_data` to pass to the `Function`

We should also define what an algorithm should give us in return!

1. The best position at the end of the minimization, $x_\text{best}$
2. The function value at that point, $f(x_\text{best})$
3. The number of function/gradient evaluations
4. Some indication as to whether the result of the minimization is valid
5. Some `String` message that can tell us any additional information about how the fit progressed/is progressing

Let's call this struct `Status` and define it as follows:
```rust
#[derive(Debug, Default, Clone)]
pub struct Status<T> {
    pub message: String,
    pub x: Vec<T>,
    pub fx: T,
    pub n_f_evals: usize,
    pub n_g_evals: usize,
    pub converged: bool,
}
impl<T> Status<T> {
    pub fn update_message(&mut self, message: &str) {
        self.message = message.to_string();
    }
    pub fn update_position(&mut self, pos: (Vec<T>, T)) {
        self.x = pos.0;
        self.fx = pos.1;
    }
    pub fn set_converged(&mut self) {
        self.converged = true;
    }
    pub fn inc_n_f_evals(&mut self) {
        self.n_f_evals += 1;
    }
    pub fn inc_n_g_evals(&mut self) {
        self.n_g_evals += 1;
    }
}
impl<T> Display for Status<T>
where
    T: Debug + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MSG:       {}", self.message)?;
        writeln!(f, "X:         {:?}", self.x)?;
        writeln!(f, "F(X):      {}", self.fx)?;
        writeln!(f, "N_F_EVALS: {}", self.n_f_evals)?;
        writeln!(f, "N_G_EVALS: {}", self.n_g_evals)?;
        write!(f, "CONVERGED: {}", self.converged)
    }
}
```

Note that we have set this up in a way that doesn't let any algorithm decrement the number of function/gradient evaluations. Additionally, no outside function can un-converge a converged `Status`. We will also typically update $f(x_\text{best})$ every time we update $x_\text{best}$, so there's only one way to do this to ensure they don't get out of sync for any reason.

Next, the `Algorithm` trait itself:

```rust
pub trait Algorithm<T, U, E> {
    fn initialize(
        &mut self,
        func: &dyn Function<T, U, E>,
        x0: &[T],
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E>;
    fn step(
        &mut self,
        i_step: usize,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E>;
    fn check_for_termination(
        &mut self,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<bool, E>;
    fn get_status(&self) -> &Status<T>;
    fn postprocessing(
        &mut self,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        Ok(())
    }
}
```
Most of these methods are fairly self-explanatory and have very similar signatures. Finally, let's wrap all of this up in a nice interface for the end-user to work with:

```rust
pub struct Minimizer<T, U, E, A>
where
    A: Algorithm<T, U, E>,
{
    pub status: Status<T>,
    algorithm: A,
    bounds: Option<Vec<Bound<T>>>,
    max_steps: usize,
    dimension: usize,
    _user_data: PhantomData<U>,
    _error: PhantomData<E>,
}

impl<T, U, E, A: Algorithm<T, U, E>> Minimizer<T, U, E, A>
where
    T: Float + FromPrimitive + Debug + Display + Default,
{
    const DEFAULT_MAX_STEPS: usize = 4000;
    pub fn new(algorithm: A, dimension: usize) -> Self {
        Self {
            status: Status::default(),
            algorithm,
            bounds: None,
            max_steps: Self::DEFAULT_MAX_STEPS,
            dimension,
            _user_data: PhantomData,
            _error: PhantomData,
        }
    }
    pub fn with_algorithm(mut self, algorithm: A) -> Self {
        self.algorithm = algorithm;
        self
    }
    pub const fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
    pub fn minimize(
        &mut self,
        func: &dyn Function<T, U, E>,
        x0: &[T],
        user_data: &mut U,
    ) -> Result<(), E> {
        self.algorithm
            .initialize(func, x0, self.bounds.as_ref(), user_data)?;
        let mut current_step = 0;
        while current_step <= self.max_steps
            && !self
                .algorithm
                .check_for_termination(func, self.bounds.as_ref(), user_data)?
        {
            self.algorithm
                .step(current_step, func, self.bounds.as_ref(), user_data)?;
            current_step += 1;
        }
        self.algorithm
            .postprocessing(func, self.bounds.as_ref(), user_data)?;
        let mut status = self.algorithm.get_status().clone();
        if current_step > self.max_steps && !status.converged {
            status.update_message("MAX EVALS");
        }
        self.status = status;
        Ok(())
    }
}
```

For now, we will ignore the `Bound` struct mentioned here, since we won't use it till we get to the `L-BFGS-B` algorithm. Note that `PhantomData` is required here because we don't actually store anything of type `U` or `E` but we need to include it in generics.
