# High Performance Computing for Cosmological Simulation Analysis

### Efficient Scientific Computing for Large-Scale Astrophysical Data

This project explores **High Performance Computing (HPC) techniques for the analysis of large-scale cosmological simulation data**, using the IllustrisTNG TNG50-1 simulation as the scientific workload.

The analysis focuses on a Milky Way–like galaxy (Subhalo 419618) and examines how computational methods can be used to extract spectral and entropy-based information from particle simulations. The project combines **scientific computing, computational astrophysics, Fourier analysis, thermodynamics, and performance optimization**.

The primary emphasis is on the **computational challenges of processing large particle datasets efficiently** rather than on astrophysical interpretation alone.

---

## Project Scale

The analysis covers **99 simulation snapshots**, processing approximately **0.21 billion particle records** in total.

Particle data are converted to a regular (128^3) spatial representation before performing three-dimensional Fourier analysis. The computational workload has a complexity of approximately

[
O(N + M\log M)
]

where (N) represents the number of particles and (M) the number of grid cells.

This provides a practical example of how computational requirements grow when working with high-resolution scientific simulations.

---

## High Performance Computing

A major component of the project is **performance profiling and optimization**.

Initial benchmarking showed that particle-to-grid deposition was the dominant computational cost, accounting for approximately **92% of the total runtime**. In comparison, the FFT and entropy calculations represented a much smaller fraction of the execution time.

This profiling helped identify where optimization would have the greatest impact.

### Performance Optimization

A caching strategy was implemented to avoid repeatedly performing the expensive grid-deposition operation.

| Metric                   | Before Optimization | After Optimization |
| ------------------------ | ------------------: | -----------------: |
| Runtime for 99 snapshots |              ~264 s |              ~20 s |
| Improvement              |                   — |           **~13×** |

The optimization demonstrates an important HPC principle:

> **Efficient scientific computing is often less about optimizing every operation and more about identifying and eliminating the dominant source of computational cost.**

The work also considers memory requirements associated with multidimensional scientific arrays, with the primary velocity-grid representation requiring roughly **25 MB per snapshot**.

---

## Computational Astrophysics

The astrophysical component provides a realistic scientific workload for the HPC analysis.

The project uses particle data from **IllustrisTNG** to study the evolution of a galaxy through its different simulation snapshots. Three-dimensional velocity fields are analyzed in Fourier space to investigate how kinetic energy is distributed across spatial scales.

The resulting power spectrum is examined through relationships of the form

[
E(k)\propto k^a
]

where the spectral exponent provides information about the scale dependence of the energy distribution.

This connects computational methods with physical questions concerning **galaxy formation, hydrodynamic flows, energy distribution, and large-scale structure**.

---

## Entropy & Thermodynamics

The project also investigates entropy from two different perspectives.

### Spectral Entropy

The normalized velocity power spectrum is used to calculate a Shannon-entropy-based measure:

[
S_H=-\sum_k p(k)\ln p(k)
]

where (p(k)) represents the normalized distribution of spectral energy.

This provides a way of quantifying the distribution of kinetic energy across spatial scales.

### Thermodynamic Entropy

A thermodynamic entropy measure is also derived from temperature and density:

[
S_T=k_B\ln\left(\frac{T^{3/2}}{n}\right)
]

Comparing (S_H) and (S_T) provides an opportunity to distinguish **kinematic/spectral organization from thermal evolution**.

The analysis reports a Pearson correlation of approximately **0.28**, suggesting that the two measures contain complementary information rather than describing identical behaviour.

---

## Key HPC Concepts

This project provides practical experience with:

* High Performance Scientific Computing
* Computational complexity
* Runtime profiling
* Performance bottleneck analysis
* Memory-aware computation
* Caching and reuse of intermediate results
* Large-scale particle data processing
* 3D numerical arrays
* Fast Fourier Transform computation
* Scientific benchmarking
* Computational astrophysics

---

## Scientific Domains

The project sits at the intersection of several areas:

**High Performance Computing**
**Scientific Computing**
**Computational Astrophysics**
**Cosmology**
**Hydrodynamics**
**Fourier & Spectral Analysis**
**Thermodynamics**
**Statistical Mechanics**
**Information Entropy**

---

## Future Work

Possible extensions include:

* Parallel processing of multiple simulation snapshots
* GPU acceleration
* Higher-resolution computational grids
* Distributed processing for larger simulation datasets
* More efficient memory management
* Analysis across multiple galaxies and subhalos
* Scaling experiments to larger scientific workloads

The long-term goal is to investigate how HPC techniques can make increasingly large cosmological simulation analyses practical while retaining the physical detail required for scientific research.

---

## Author

**Shaurya Pratap Singh Yadav**
M.Tech. — AI for Sustainability
Indian Institute of Technology Kanpur

---

### Project Focus

> **Using High Performance Computing to turn large-scale cosmological simulation data into computationally tractable scientific insight.**
