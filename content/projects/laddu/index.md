+++
+++
<!-- markdownlint-disable MD034 -->
{{ image(url="https://github.com/denehoffman/laddu", url_min="laddu_logo.png", no_hover=true) }}

[`laddu`](https://github.com/denehoffman/laddu) (/ˈlʌduː/) is a library for analysis of particle physics data. It is intended to be a simple and efficient alternative to some of the other tools out there. `laddu` is written in Rust with bindings to Python via `PyO3` and `maturin` and is the spiritual successor to `rustitude`, one of my first Rust projects. The goal of this project is to allow users to perform complex amplitude analyses (like partial-wave analyses) without complex code or configuration files.

`laddu` grew out of my frustration with the way amplitude analyses were being done within the GlueX collaboration. Everyone had these messy configuration files which would need to be duplicated and modified, usually by one-off scripts, to produce fit results which would then have to be collected by yet another set of scripts. I got tired of the constant file management, I was spending more time debugging config files than actually doing physics! Since my original foray into Rust in March 2024, I have learned a lot about what is required to distribute a project like this via Python. There are tons of small optimizations that can be made, and I'd imagine there are still quite a few to go. This project has taught me everything from memory management to the intricacies of floating-point numbers to quite a lot about parallel processing. I believe the project is still in an exploratory state, but it is certainly usable enough to do some actual research now. Since I'm actively using it while I develop it, I quickly discover new sharp corners and quality-of-life features to implement, and there's always little chores to do like documentation and testing.

Here's a short demo of `laddu` in action:

{{ video(url="laddu_demo.webm", controls=true) }}
