# Math framework

## docs/theory.md
## Mathematical Theory

## Core Equations

### Phase Evolution
\[
\theta_L(t+1) = \theta_L(t) + \Delta_L \mod 2\pi
\]
\[
\theta_R(t+1) = \theta_R(t) + \Delta_R \mod 2\pi
\]

### Symmetry Operations
For a kaleidoscope with $n$ mirrors:
\[
S(\theta, F) = \bigcup_{i=1}^{N} \bigcup_{k=0}^{n-1} R\left(\theta + \frac{2k\pi}{n}\right) f_i
\]

## Parameter Space

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Left phase | $\theta_L$ | $[0, 2\pi)$ | Left perceptual framework |
| Right phase | $\theta_R$ | $[0, 2\pi)$ | Right perceptual framework |
| Phase diff | $\Delta\theta$ | $[0, \pi]$ | Perceptual dissonance |
| Symmetry folds | $n$ | $\mathbb{N}^+$ | Cognitive complexity |
| Fragments | $N$ | $\mathbb{N}^+$ | Sensory data points |
