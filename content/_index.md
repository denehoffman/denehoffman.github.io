+++
[extra]
scripts = ["home-links.js"]
+++

# Dr. Dene Hoffman

<aside>
  {{ image(url="prof_pic.jpg", alt="Portrait of Dene Hoffman", no_hover=true) }}
</aside>

{{ project_grid(featured=true) }}

{% home_section(kind="research") %}
<svg class="home-research__motif" viewBox="0 0 200 112" aria-hidden="true">
  <path class="home-research__photon" d="M4 32c5-7 10 7 15 0s10 7 15 0 10 7 15 0 10 7 15 0" />
  <path d="M4 82h64M64 32c28-2 48-3 72-2M68 82c43 3 84 10 128 18M136 30c20-1 34-11 60-20M136 30c22 3 37 14 60 25" />
  <path class="home-research__exchange" d="M64 32c14 15 15 35 4 50" />
  <circle cx="64" cy="32" r="3" />
  <circle cx="68" cy="82" r="3" />
  <circle cx="136" cy="30" r="3" />
</svg>
<h2>Postdoctoral Research Associate studying glueballs at GlueX</h2>

<div class="home-research__glance" aria-label="Research at a glance">
  <p><span>Experiment</span><strong>GlueX</strong></p>
  <p><span>Method</span><strong>Amplitude analysis</strong></p>
  <p><span>Physics</span><strong>Exotic hadrons</strong></p>
  <p><span>Tools</span><strong>Scientific Rust</strong></p>
</div>

I'm currently a postdoctoral researcher at The College of William & Mary studying the strong force through the GlueX collaboration. GlueX is a multinational collaboration located in Hall D at Jefferson Lab which collides high-energy photons with a proton target.

My current work consists of a study of $`K_SK_S`$ (pairs of K-short mesons) photoproduction. This gives us access to even-spin $`f`$ and $`a`$ mesons (light, flavorless particles with isospin $`0`$ and $`1`$ respectively), the former of which are interesting for several reasons. First, they share many of the same quantum numbers, the values we use to classify particles, with glueballs, hypothetical particles that contain only gluons, the force-carrier of the strong interaction. The lightest of these glueballs is predicted to look nearly identical to some of the $`f_0`$ mesons (spin-$`0`$ $`f`$ mesons), and it turns out that there are too many of these $`f_0`$ mesons seen in experiments for them to all be compatible with the quark model.

[Read more about my research.](@/research/index.md)
{% end %}

{% home_section(kind="writing") %}
<div class="home-section__heading">
  <h1>Latest writing</h1>
  <a href="/blog/">All posts</a>
</div>
<div class="home-writing">
  <a class="home-writing__post" href="/blog/ganesh-a-new-optimization-crate-for-rust/">
    <time datetime="2025-09-08">September 08, 2025</time>
    <strong>ganesh: A New Optimization Crate for Rust</strong>
    <span>A different approach to optimization in Rust</span>
  </a>
  <a class="home-writing__post" href="/blog/the-bfgs-algorithm-family-in-rust-part-3/">
    <time datetime="2025-06-18">June 18, 2025</time>
    <strong>The BFGS Algorithm Family in Rust (Part 3)</strong>
    <span>The L-BFGS-B implementation</span>
  </a>
  <a class="home-writing__post" href="/blog/tzigane/">
    <time datetime="2025-06-16">June 16, 2025</time>
    <strong>Tzigane</strong>
    <span>Performance by me with the All University Orchestra</span>
  </a>
</div>
{% end %}

{% home_section(kind="group") %}
# Research Group
<div class="home-roster">
  <p><a href="https://kuessner.gitlab.io"><strong>Dr. Meike Küßner</strong></a><span>Principal Investigator</span></p>
  <p><strong>Addison Kovats-Bernat</strong><span>Graduate Student</span></p>
</div>
{% end %}

{% home_section(kind="links") %}
# Links

<div class="home-links">
  <a class="home-link home-link--email" href="#contact" data-contact-user="nhoffman" data-contact-host="wm" data-contact-domain="edu">
    <span class="home-link__icon" aria-hidden="true">@</span>
    <span><strong>Email me</strong><small>Get in touch</small></span>
  </a>
  <a class="home-link" href="https://orcid.org/0000-0002-8865-2286">
    <span class="home-link__icon home-link__icon--orcid" aria-hidden="true">iD</span>
    <span><strong>ORCID</strong><small>0000-0002-8865-2286</small></span>
  </a>
  <a class="home-link" href="https://scholar.google.com/citations?user=39-XmFUAAAAJ">
    <span class="home-link__icon home-link__icon--scholar" aria-hidden="true">
      <svg viewBox="0 0 24 24" role="img"><path d="M12 24a7 7 0 1 0 0-14 7 7 0 0 0 0 14Zm0-24L0 7.5l4.38 2.63A9.97 9.97 0 0 1 12 6c3.14 0 5.94 1.45 7.78 3.72L24 7.5Z"/></svg>
    </span>
    <span><strong>Google Scholar</strong><small>Publications and citations</small></span>
  </a>
  <a class="home-link" href="https://github.com/denehoffman">
    <span class="home-link__icon home-link__icon--github" aria-hidden="true">
      <svg viewBox="0 0 24 24" role="img"><path d="M12 .3a12 12 0 0 0-3.79 23.39c.6.11.82-.26.82-.58v-2.24c-3.34.73-4.04-1.42-4.04-1.42-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 1.21.08 1.84 1.24 1.84 1.24 1.07 1.84 2.81 1.31 3.5 1 .11-.78.42-1.31.76-1.61-2.66-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.11-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.29-1.55 3.3-1.23 3.3-1.23.65 1.66.24 2.88.12 3.18a4.63 4.63 0 0 1 1.23 3.22c0 4.61-2.81 5.62-5.48 5.92.43.37.81 1.1.81 2.22v3.29c0 .32.22.7.83.58A12 12 0 0 0 12 .3Z"/></svg>
    </span>
    <span><strong>GitHub</strong><small>Code and projects</small></span>
  </a>
</div>
<noscript><p class="home-email-fallback">Email: nhoffman at wm dot edu</p></noscript>
{% end %}
