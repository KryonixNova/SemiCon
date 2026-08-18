# Public sources

Citations backing this project's synthetic-data and augmentation design
choices, per the hackathon spec's requirement to justify structures, noise,
and augmentations against credible public sources.

## DRAM 1T-1C cell structure (word lines, bit lines, capacitor storage)

- imec, "DRAM peripheral transistors technology platform."
  <https://www.imec-int.com/en/articles/technology-platform-thermally-stable-dram-peripheral-transistors>
  Describes the 1-transistor/1-capacitor DRAM cell and its access-transistor
  role, backing `src/patterns/dram.py`'s word-line/bit-line/contact/
  capacitor layout model.
- SemiAnalysis, "The Memory Wall: Past, Present, and Future of DRAM."
  <https://newsletter.semianalysis.com/p/the-memory-wall>
  Industry-level explanation of the 1T-1C array structure (word lines
  driving access-transistor gates, bit lines carrying the sensed charge)
  that this project's `cell_pitch_nm`/`word_line_*`/`bit_line_*` parameters
  model as free, non-proprietary parameters (no specific process node is
  implied -- see `generate_dataset.py`'s own module docstring language,
  matching AFB's precedent).

## SEM imaging noise and degradation modeling

- "Correction of Scanning Electron Microscope Imaging Artifacts in a Novel
  Digital Image Correlation Framework," *Experimental Mechanics*
  (Springer), open access via PMC.
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC6541586/>
  Documents drift/charging-driven image distortion in real SEM capture --
  directly motivates `src/sem_imaging.py`'s `apply_raster_drift`,
  `add_charging_streaks`, and the `drift_jitter_px`/`shear_amplitude_px`
  parameters this project's `IMAGING_NOISE_PROFILES` randomizes.
- "Scanning Electron Microscope Image Signal-to-Noise Ratio Monitoring for
  Micro-Nanomanipulation," open access via HAL.
  <https://hal.science/hal-01051309/document>
  Establishes that SEM pixel noise is dominated by Poisson-distributed shot
  noise from the primary/secondary electron count, plus detector/amplifier
  noise -- the physical basis for this project's `dose`-driven shot-noise
  simulation and `detector_noise_sigma` parameter in `sem_imaging.py`.

## Data augmentation for scale/rotation robustness in matching tasks

- "An Efficient Deep Template Matching and In-Plane Pose Estimation Method
  via Template-Aware Dynamic Convolution," arXiv.
  <https://arxiv.org/html/2510.01678>
  Uses rotation/shear-based augmentation during training so a deep template
  matcher regresses position, rotation, and scale robustly -- the same
  training-time-augmentation strategy (rather than inference-time
  multi-hypothesis search) this project's `GEOMETRIC_PROFILES["drift"]`
  applies to reference crops.
- "Who Handles Orientation? Investigating Invariance in Feature Matching,"
  arXiv. <https://arxiv.org/html/2604.11809v1>
  Finds that rotation robustness in learned feature matchers can emerge
  from training-distribution diversity (with or without explicit rotation
  augmentation), supporting this project's design choice to bake
  scale/rotation robustness into the training distribution rather than
  adding an inference-time geometric search (which would cost runtime,
  directly graded by the spec).
