+++
+++

# Dr. Dene Hoffman

<aside>
  {{ image(url="prof_pic.jpg", no_hover=true) }}
</aside>

<nav class="home-projects" aria-label="Featured projects">
  <p>Featured projects</p>
  <div>
    <a class="home-project home-project--featured" href="/projects/laddu/">
      <strong>laddu</strong>
      <span>Amplitude analysis · Python + Rust</span>
    </a>
    <a class="home-project" href="/projects/ganesh/">
      <strong>ganesh</strong>
      <span>Optimization · Rust</span>
    </a>
    <a class="home-project" href="https://github.com/denehoffman/maryada">
      <strong>maryada</strong>
      <span>GitHub repository</span>
    </a>
    <a class="home-project" href="https://github.com/denehoffman/gluex-rs">
      <strong>gluex-rs</strong>
      <span>GitHub repository</span>
    </a>
  </div>
</nav>

{% home_section(kind="research") %}
## Postdoctoral Research Associate studying glueballs at GlueX

I'm currently a postdoctoral researcher at The College of William & Mary studying the strong force through the GlueX collaboration. GlueX is a multinational collaboration located in Hall D at Jefferson Lab which collides high-energy photons with a proton target.

The main goal of GlueX is a search for particles with exotic quark content. Standard composite particles are identified in two main categories, mesons (made of a quark-antiquark pair) and baryons (three quarks). However, there has been recent experimental evidence of states with more than three quarks, dubbed tetraquarks, pentaquarks, etc. Computer simulations (Lattice QCD) predict additional states such as glueballs, which contain no quarks at all and are just bound states of the "gluons" that hold matter together, and hybrid mesons, where a valence gluon contributes to the total angular momentum to produce "forbidden" quantum numbers.

My current work consists of a study of $`K_SK_S`$ (pairs of K-short mesons) photoproduction. This gives us access to even-spin $`f`$ and $`a`$ mesons (light, flavorless particles with isospin $`0`$ and $`1`$ respectively), the former of which are interesting for several reasons. First, they share many of the same quantum numbers, the values we use to classify particles, with glueballs, hypothetical particles that contain only gluons, the force-carrier of the strong interaction. The lightest of these glueballs is predicted to look nearly identical to some of the $`f_0`$ mesons (spin-$`0`$ $`f`$ mesons), and it turns out that there are too many of these $`f_0`$ mesons seen in experiments for them to all be compatible with the quark model.

However, the downside of this particular study is that there are lots of other particles present in the $`K_SK_S`$ channel, and many of them overlap each other. We are working on coupling this channel with others that can constrain some of the states.

Hopefully my work at GlueX can provide a small step in disentangling this complex set of states and move us closer to understanding the complex physics of quantum chromodynamics. The entire thesis, along with the analysis code and results (including some not included in the thesis) can be found [here](https://github.com/denehoffman/thesis). The thesis is also available on [CMU's thesis repository](https://kilthub.cmu.edu/articles/thesis/Photoproduction_of_K_sup_0_sup_sub_S_sub_Pairs_at_GlueX/29950307).

I recently defended my thesis and am now focused on some software projects related to that work, particularly [`laddu`](https://github.com/denehoffman/laddu), an amplitude analysis library for Python and Rust, and [`ganesh`](https://github.com/denehoffman/ganesh), an optimization library written in pure Rust.
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
