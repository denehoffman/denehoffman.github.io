# bubble chamber realism: research and practical options

Research date: 10 November 2026. This records the research and the first implementation pass; it is not a calibrated detector model. Existing site changes and the unfinished post are outside this note's scope.

## what the photographs actually record

The particle trajectory, the visible bubbles, and the recorded photograph have different time scales. CERN describes particles traversing metres of liquid in nanoseconds, with visible bubble growth taking milliseconds. The operating sequence is expansion, beam passage, bubble growth, flash photography, and recompression. Slow trajectory playback remains useful, but it represents an explanatory replay rather than live optical observation. [CERN teaching material](https://indico.cern.ch/event/724025/contributions/3380808/attachments/1831550/2999589/TP_BubbleChamber.pdf), [CERN archived explanation](https://hst-archive.web.cern.ch/archiv/hst2000/teaching/resource/bubble/bubble.htm).

A useful reference is [CERN's 1973 BEBC photograph](https://repository.cern/records/j95hs-s5y38). Its caption identifies a parallel negative-kaon beam, unrelated cosmic-ray and externally produced tracks, a piston surrounded by bubbles, and valves and windows. The image was inspected at full resolution during this research. It shows uneven background texture, pointlike marks, several track widths and contrasts, and substantial fixed apparatus structure. A ring, dot, or cross must not automatically be interpreted as a particle vertex. Those observations motivate separate particle, chamber, and photographic layers rather than one undifferentiated noise field.

## recommended implementation order

| priority | change | feasible approach | effort |
| --- | --- | --- | --- |
| 1 | individual bubbles instead of randomly broken strokes | Sample persistent bubble positions along path length; render small discs or shaded points with modest size variation. Keep the same samples as the animation advances and during PNG export. Use a low-detail aggregate renderer for distant/subpixel bubbles. | small–medium |
| 2 | a background with causes | Generate a sparse population of long unrelated cosmic/external tracks, a correlated group of nearly parallel beam tracks, low-density isolated marks, and a few persistent nucleation patches near a chamber boundary. Give these independent rates and visibility. | medium |
| 3 | chamber-wide exposure timing | Offer an optional photograph mode: reveal an entire event through bubble growth, hold the frame, then clear globally through recompression. Keep the existing slow-motion replay as another view. A flash should be a restrained exposure change, not a bright strobe. | medium |
| 4 | a declared liquid and material model | Begin with one hydrogen preset, specifying density, radiation length, and stopping-power tables; add a dense-liquid preset only with consistent data. Keep display speed, magnification, and decay exaggeration as separate presentation settings. | medium |
| 5 | physical roughness and stopping | Add angular scattering increments with the correct path-length scaling; update momentum from energy loss; let stopping tracks become denser and tighter where appropriate. Subdivide highly curved sections. | medium |
| 6 | secondary structures | Add rare, connected delta-electron curls; then bremsstrahlung and displaced photon conversions. Record the interaction vertex and energy transfer so this does not become random decorative branching. | medium–large |
| 7 | depth and photographic response | Integrate trajectories in 3D and project them through a camera; vary blur and contrast with depth. Keep bubbles off the exact mathematical centreline only through a small fixed measurement/optical spread. Add weak static grain, illumination variation, and optional fiducials. | large for 3D; small for restrained optics |
| 8 | more varied interaction histories | Add material scattering, recoil tracks, hadronic interaction vertices, and richer decay sequences only after transport and medium choices are coherent. Distinguish a decay from a collision in the event record. | large |

The table contains implementation recommendations, not experimentally measured parameter choices. In particular, a Poisson process is a useful first rendering model for sampled bubble locations, not a claim that all chambers have independent, uniform nucleation.

## particle physics that should drive the image

PDG's material-interaction review supplies the relevant dependencies: charged-particle stopping depends on velocity and material; losses fluctuate; many small Coulomb deflections produce multiple scattering with non-Gaussian hard tails; energetic knock-on electrons make secondary tracks. Electrons and positrons require their own stopping treatment, including bremsstrahlung, while photons can produce displaced pairs. These mechanisms justify density changes, coherent roughness, and occasional connected branches. [PDG, passage of particles through matter](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-passage-particles-matter.pdf), especially sections 34.2.3, 34.2.7, 34.2.9, 34.3, and 34.4.

For implementation, use scattering increments whose variance scales with material path length; do not add equally sized independent angle kicks at every integration step. Otherwise changing the transport resolution changes the apparent physics. Likewise, sample bubble density per unit physical path length, rather than once per rendered segment. Use the projected curvature only with awareness that a camera image need not show the circle in the plane perpendicular to the magnetic field.

## how to add noise without making the wrong detector

A clean chamber is not continuously boiling everywhere. Alvarez's original report explicitly describes clean hydrogen chambers in which ionizing particles are needed to initiate boiling. Glaser's Nobel lecture and CERN's historical account describe the importance of apparatus design and timing. Hence background bubbling should be controlled and structured, not a default snowstorm. [Alvarez, liquid hydrogen bubble chambers (1956)](https://digital.library.unt.edu/ark:/67531/metadc890458/), [Glaser, Nobel lecture](https://www.nobelprize.org/uploads/2017/05/glaser-lecture.pdf), [CERN Courier historical account](https://cern-courier.web.cern.ch/a/when-the-bubble-chamber-first-burst-onto-the-scene/).

Suggested layers:

- **Physical:** unrelated cosmic/external tracks, beam contamination, secondary particles, and sparse nucleation. Their locations persist within an exposure.
- **Apparatus:** stable edge structures, occasional local boiling patches, optical windows, and optional calibration marks. These are not extra particles.
- **Photographic:** weak fixed grain, uneven illumination, focus variation, and exposure/contrast response. A filmed negative can be displayed with either polarity; light/dark website modes need not imply different particle physics.

Avoid per-frame positional jitter, rapidly flickering random specks, uniformly rising smoke, or a luminous particle head as supposed photographic realism. Those obscure the difference between transport, bubble formation, and image acquisition. Slight global image motion can be offered as a stylistic effect, but it should not perturb conserved trajectories or continuously shake individual bubbles.

## calibration limits and practical scope

BEBC used hydrogen, deuterium, and neon–hydrogen mixtures; its operating sensitivity was controlled through expansion. Different chamber media and operating points should not share one invented universal noise rate, bubble density, interaction length, or scattering strength. [CERN BEBC description](https://repository.cern/records/5t6ea-6k552).

The initial code audit found a physical curvature relation and decay kinematics. Its unconfigured material losses, purely planar transport, per-segment texture, and conditioned visible decay sampling are where a more realistic model would pay off. In particular, the audited constants are 0.003 GeV/m for heavier particles and 0.055 GeV/m for electrons (about 18 times larger), without a declared liquid. Those are toy parameters, not a coherent material model; replace them with species- and energy-dependent stopping before describing any preset as calibrated. Preserve useful physics and improve the image-formation layer first. A full Geant4-style simulation or fluid-dynamics solver is unnecessary for a convincing browser toy.

A sensible first tranche is individual bubbles, structured background tracks/marks, a coherent exposure lifecycle, and a material preset. A second tranche adds calibrated scattering, stopping, and delta rays. A third adds 3D projection and richer secondary interactions.

## source access notes

The CERN repository photograph and its original caption, the BEBC description, the Alvarez report description, and PDG text were accessible. Some older CERN teaching pages now redirect to SSO; their indexed text and CERN-hosted teaching PDF still supply the cycle/timing explanation. The Glaser Nobel PDF was discoverable with indexed text but its direct fetch returned 403. No claim here depends on an invented exact background rate or a bubble-growth coefficient extracted from inaccessible material.

## baseline audit before the realism pass

The original `static/bubble-chamber.js` audit identified these starting points:

- `transportCharged` uses the usual momentum–curvature relation and reduces momentum as kinetic energy is deposited. The magnetic rotation itself should continue to do no work.
- `twoBodyDecay` and `threeBodyDecay` already check four-momentum conservation in their planar model. Preserve those checks as transport and secondary interactions become richer.
- `addTexture` currently samples missing line segments and fixed species-dependent stroke widths. Replace this image layer with persistent bubble samples independent of transport step size.
- Stopping power is currently an unconfigured constant multiplied by a capped inverse-beta-squared factor, with a separate much larger electron constant. There is no defined liquid, radiation length, scattering model, or radiative-loss model; these constants should not be interpreted as a calibrated medium.
- Transport uses fixed 3.5-pixel steps. Use adaptive steps or analytic magnetic arcs so a tight low-momentum curl cannot turn substantially in one coarse step.
- Decay positions are sampled conditional on occurring inside the viewport, and physical decay lengths are multiplied by eight. These are presentation choices. A realism preset should sample unconditioned lifetimes and tolerate particles escaping without a visible decay, while the existing event explorer can keep conditioning as an explicitly selected demonstration.
- Decays and trajectories are planar. A later 3D model would use full momentum vectors, propagate helices, and project them through a camera. This should also determine the apparent cosmic-ray direction; “downward” in the lab need not mean downward on the screen.

The CERN 2 m teaching photographs were also inspected directly: [photo 1](https://scienceinschool.org/wp-content/uploads/2019/03/issue46_bubblechamber_figure2.jpg), [photo 2](https://scienceinschool.org/wp-content/uploads/2019/03/issue46_bubblechamber_figure4.jpg). Their colored curves are instructional overlays. They show beam bundles, fine crossing tracks and curls, scattered background marks, and fixed crosses/border markings. Do not copy the overlay colors as an intrinsic detector response.

Before describing a mode as physically calibrated, verify transport step-size convergence, momentum and charge accounting at generated interactions (including medium recoil), kinetic-energy conservation with losses disabled, ranges against the chosen material tables, and the expected lifetime distribution without visibility conditioning. Rendering seeds should make a paused image and its export reproducible.

## current implementation

`static/bubble-physics.js` implements adaptive planar transport in liquid hydrogen: density 0.0708 g/cm³, radiation length 8.905 m, mean excitation energy 21.8 eV, and the corresponding Sternheimer density-effect parameters. Heavy particles use Bethe stopping; electrons and positrons use their respective collision-loss expressions. A low-velocity continuation regularizes stopping below the useful Bethe regime. Angular diffusion approximates multiple scattering. [PDG material page](https://pdg.lbl.gov/2017/AtomicNuclearProperties/HTML/liquid_hydrogen.html), [PDG muon table](https://pdg.lbl.gov/2017/AtomicNuclearProperties/MUE/muE_liquid_hydrogen.txt).

The playground exposes view, reaction, activity, and replay speed, with direction, density, and a random-angle switch in a collapsed Beam section. Random angle picks one direction per passage, shows that direction in the existing aim control, and restores the previous manual aim when switched off; it never rotates tracks already present. Reaction selection applies to both manual and automatic events. There are no scene presets, independent background sliders, manual track-retention controls, or photographic exposure timing controls. The background keeps its intentional fixed edge clusters and isolated specks. “Beam burst” is used in the interface; “spill” is valid accelerator terminology, but this toy does not simulate extraction. [CERN beam guide](https://beamline-for-schools.web.cern.ch/sites/default/files/BL4S-Beam-and-detectors_2018_0.pdf).

Automatic reactions belong to beam passages. `static/bubble-interactions.js` generates on-shell pion–proton production at the actual transported beam momentum, including a stationary hydrogen proton. Each reacting primary terminates at the chosen production point. Elastic scattering continues with a deflected pion and recoil proton; inelastic reactions replace the primary with a complete set of associated products. The selected channels are:

- track: π⁻p → π⁻p;
- decay: π⁻p → K⁰Λ;
- cascade: π⁻p → K⁺K⁰Ξ⁻;
- Dalitz: π⁻p → π⁻pπ⁰, followed by π⁰ → γe⁺e⁻.

Production checks energy, both planar momentum components, charge, baryon number, and strangeness, and rejects inaccessible thresholds. Two- and three-body production use Lorentz boosts and factorized phase space; directions remain planar and are not fitted angular distributions. Vertices are selected from the central 15–70% of the visible primary path for framing, not sampled from a measured nuclear interaction length. [PDG kinematics](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-kinematics.pdf). K⁰Λ production is an established pion–proton channel; the listed weights are illustrative, not extracted cross sections. [Experimental measurement](https://arxiv.org/abs/2505.18598).

Secondary transport starts from these generated four-vectors. Visible Ξ, Λ, K⁰, and π⁰ decays conserve local four-momentum at their actual decay momentum, including charged-parent energy loss before decay. Neutral photons remain in the event record without drawing tracks. Neutral-kaon propagation demonstrates the short-lived component and selected charged decay channels rather than a full K⁰–K̄⁰ mixing/branching simulation. Lifetime magnification and conditioning on a visible decay remain presentation choices. Long-lived charged pions and associated charged kaons are transported through the view without rare weak decays.

Only unrelated cosmic tracks are generated independently. Manual clicks preserve the existing selected topology demonstrations at arbitrary positions; they are not represented as complete beam–target collisions. Activity is mean reactions per passage (0–10), capped at one reaction per primary and by the display track budget. A beam event reserves up to 80 tracks, leaving room below the global 96-track cap for manual input. All incoming prefixes and associated outgoing tracks are prepared together, so cancellation cannot leave a reaction without its incoming particle.

The bunch has longitudinal phase spread equivalent to a half-normal separation along travel (180-pixel scale, capped at 520 pixels). This is represented as entry-time offsets at each particle’s velocity, keeping the entry port and collimation unchanged rather than adding fictitious upstream magnetic bending. Particles therefore enter over a short interval rather than forming a common moving front. Interaction and decay delays include each primary's actual transport time to its vertex; photograph mode continues to show the developed exposure together.

A new beam waits until the previous beam event, including its associated products, has completely faded. Replay retention adapts to estimated incoming track load, aiming for roughly 40 visible tracks; a hard 96-track limit makes old secondary/cosmic tracks give way under repeated manual input. The active beam is protected from eviction so it completes its fade. Display speed remains a presentation setting. Photo mode fades each new exposure in for one simulation second, holds it for seven, then resets directly to the next exposure without compression or fade-out. Next photo always clears the previous exposure. Pausing holds a frame for viewing or PNG export. Generated-event toasts are omitted.

## rendering and validation

`static/bubble-renderer.js` caches tightly cropped monochrome layers in 192-pixel spatial tiles and 64 ms birth-time slices. Photograph mode needs spatial tiles only; after creating a combined image it releases temporary tiles and sample arrays. A final tint allows theme changes without rebuilding geometry. Raster storage is bounded at 96 MiB, with cached vector paths as a fallback. Removing events explicitly releases cached canvases. Preparation yields between small batches, and clear, mode changes, and disabling the beam cancel pending work.

Image formation uses three strength bands, slice-midpoint fades (at most 0.8 percentage points of alpha error for a four-second fade), and covered-area opacity to approximate subpixel bubble growth. These do not alter transported positions or fully developed bubble radii. Integration, lifetime conditioning, bubble sizes, and playback timing remain illustrative; this is not a calibrated detector. Production and decay vertices now conserve four-momentum, and the target baryon is explicitly included. This is still a small illustrative reaction catalogue, not a hadronic shower or cross-section model. Continuous transport approximates energy deposition and scattering rather than tracking every microscopic medium recoil. Delta rays, hard scattering tails, radiative electron losses, and 3D projection remain future work.

Numerical checks cover PDG muon stopping reference points, magnetic curvature and energy conservation with losses disabled, range convergence, scattering step-size invariance, and bubble-density invariance under path subdivision. Additional numerical checks exercise all production channels over many incoming directions and momenta, thresholds from either side, an analytic elastic forward-scattering limit, and the secondary decay chains. Browser checks exercise primary termination at the production vertex, recoil starts, local production balance, longitudinal entry-time spread, reaction controls, non-overlapping passages, retention limits, photo growth/reset, pause/export, accessibility, and reduced-motion preferences.

Run `npm run benchmark:chamber -- http://127.0.0.1:1111` for the current maximum beam/activity settings at 1440×900 and DPR 2. Set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` when needed. `CHAMBER_PERF_ASSERT=1` opts into a machine-dependent 16.7 ms p95 animation-callback budget. The original 120-track replay measured 317.6 ms p95 before caching; current benchmarks use the new automatic track budget and single-beam policy, so their scene differs. Callback CPU time is not an overall frame-rate guarantee.

Before conserved beam-target production was added, the maximum-setting benchmark (60 primary beam tracks, requested mean 10 reactions per passage, 96-track budget) measured p95 animation-callback CPU times of 1.3 ms in replay and 0.3 ms in photographs, with no tasks above 50 ms recorded in either six-and-a-half-second sample. These are local Chromium measurements, not a universal frame-rate promise.

The complete conserved reaction batches revealed an additional preparation bottleneck: timer-clamped yields could leave the maximum-setting beam preparing after 6.5 seconds. Private `MessageChannel` tasks now yield between the same bounded batches without nested timer waits. The isolated static-build benchmark then completed both modes with p95 callbacks of 1.0 ms (replay) and 0.2 ms (photographs), and no recorded tasks above 50 ms. Physics, bunch spread, track count, and image detail were unchanged in this comparison.
