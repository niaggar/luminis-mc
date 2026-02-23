# Luminis-MC

**Simulación Monte Carlo de transporte de fotones en medios turbios**

Librería de alto rendimiento escrita en C++ con bindings de Python, diseñada para simular dispersión múltiple de luz incluyendo polarización, coherencia y Coherent Backscattering (CBS).

---

## Características principales

- **Motor Monte Carlo vectorial** con polarización completa (matrices de Jones / parámetros de Stokes)
- **Coherent Backscattering (CBS)** — cálculo del path reverso por reciprocidad
- **Muestras multicapa** — capas con diferentes propiedades ópticas apiladas en el eje z
- **Múltiples funciones de fase** — Mie, Henyey-Greenstein, Rayleigh, Rayleigh-Debye-Gans, Draine
- **Solver de Mie completo** — via Fortran 77 (MIEV0 de Wiscombe), con tablas precomputadas
- **Detectores variados** — campo cercano/lejano, fluencia, speckle, CBS, estadísticas
- **Ejecución multi-hilo** — paralelismo real con `std::thread` y merge de detectores
- **Interfaz Python** — acceso zero-copy a resultados vía NumPy (buffer protocol)

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Núcleo computacional | C++23 |
| Solver de Mie | Fortran 77 (MIEV0) |
| Bindings Python | pybind11 |
| Build system | scikit-build-core + CMake |
| Paralelismo | `std::thread` |

---

## Instalación

### Requisitos previos

- Python 3.10+
- Compilador C++23 (GCC 13+, Clang 16+, Apple Clang 15+)
- Compilador Fortran (gfortran)
- CMake 3.15+

### Instalar desde fuente

```bash
git clone https://github.com/niaggar/luminis-mc.git
cd luminis-mc
pip install .
```

Para desarrollo (editable):

```bash
pip install -e .
```

> Internamente, `scikit-build-core` invoca CMake, que compila la librería Fortran de Mie (`mie_fortran`), el core C++ (`_core_c`) y el módulo Python (`_core`).

---

## Uso rápido

```python
import luminis_mc as lmc
import numpy as np

# 1. Función de fase y medio dispersivo
phase = lmc.MiePhaseFunction(radius=0.5e-3, n_p=1.59, n_m=1.33, wl=0.633e-3)
medium = lmc.MieMedium(mu_a=0.01, mu_s=10.0, phase=phase,
                        radius=0.5e-3, n_particle=1.59, n_medium=1.33,
                        wavelength=0.633e-3)

# 2. Muestra (agua como solvente, una capa semi-infinita)
sample = lmc.Sample(n_medium=1.33)
sample.add_layer(medium, 0.0, float('inf'))

# 3. Láser
laser = lmc.Laser(wavelength=0.633e-3, n_medium=1.33)

# 4. Detector CBS en campo lejano
sensors = lmc.SensorsGroup()
det = sensors.add_detector(lmc.FarFieldCBSSensor(0.1, 6.28, 100, 1))

# 5. Configurar y ejecutar
config = lmc.SimConfig(n_photons=1_000_000, sample=sample,
                        laser=laser, detector=sensors)
config.n_threads = 8
config.track_reverse_paths = True
lmc.run_simulation_parallel(config)

# 6. Resultados
result = lmc.postprocess_farfield_cbs(det, 1_000_000)
enhancement = np.array(result.coherent.S0) / np.array(result.incoherent.S0)
```

---

## Estructura del proyecto

```
luminis-mc/
├── CMakeLists.txt                  # Build system (CMake + scikit-build)
├── pyproject.toml                  # Metadata del paquete Python
│
├── include/luminis/                # Headers C++ (interfaz pública)
│   ├── core/                       #   Photon, Laser, Medium, Sample, Detector, Simulation
│   ├── math/                       #   Vec3, Matrix, RNG, utilidades matemáticas
│   ├── sample/                     #   Funciones de fase, tablas de muestreo, MFP
│   ├── log/                        #   Logger singleton
│   └── mie/                        #   Interfaz al solver Mie (Fortran)
│
├── src/                            # Implementaciones C++
│   ├── simulation.cpp              #   Loop principal de propagación
│   ├── detector.cpp                #   Todos los sensores (~1700 líneas)
│   ├── medium.cpp                  #   Muestreo de scattering, matrices de Mie
│   ├── phase.cpp                   #   Funciones de fase y muestreo condicional
│   └── mie/                        #   Solver de Mie (Fortran + wrapper C++)
│
├── python/
│   ├── python_module.cpp           # Bindings pybind11
│   └── luminis_mc/                 # Paquete Python
│       ├── __init__.py             #   Re-exporta todo desde _core
│       └── manager.py              #   Experiment y ResultsLoader
│
├── tests/                          # Scripts de simulación (Python)
└── apps/                           # Aplicaciones standalone C++
```

---

## Arquitectura

### Pipeline de simulación

```
Laser → emit_photon() → [Propagación Monte Carlo] → Detectores → Resultados (NumPy)
```

1. **Configuración**: se definen `PhaseFunction`, `ScatteringMedium`, `Sample`, `Laser` y `SensorsGroup`
2. **Ejecución**: `run_simulation_parallel()` distribuye fotones entre hilos; cada hilo opera con detectores clonados (sin locks)
3. **Propagación** (`run_photon`): para cada fotón se itera el ciclo *step → detección → scattering → absorción*
4. **Merge**: al finalizar, los detectores locales se fusionan en el principal
5. **Post-procesamiento**: normalización por ángulo sólido y número de fotones

### Medios dispersivos

| Clase | Descripción |
|---|---|
| `RGDMedium` | Rayleigh-Gans-Debye (partículas pequeñas, fórmula analítica) |
| `MieMedium` | Mie completo (tablas precomputadas via MIEV0, sin restricción de tamaño) |

### Funciones de fase

| Clase | Modelo |
|---|---|
| `UniformPhaseFunction` | Isotrópica |
| `HenyeyGreensteinPhaseFunction` | Parametrizada por $g$ |
| `RayleighPhaseFunction` | Rayleigh |
| `RayleighDebyePhaseFunction` | Rayleigh-Debye-Gans |
| `RayleighDebyeEMCPhaseFunction` | RDG con corrección EMC |
| `DrainePhaseFunction` | Draine ($g$, $\alpha$) |
| `MiePhaseFunction` | Mie exacta (tablas precalculadas) |

### Detectores

| Clase | Uso |
|---|---|
| `PhotonRecordSensor` | Snapshot individual por fotón (post-procesamiento en Python) |
| `PlanarFieldSensor` | Campo eléctrico complejo en plano $(x, y)$ — speckle |
| `PlanarFluenceSensor` | Stokes en plano $(x, y, t)$ — fluencia espaciotemporal |
| `FarFieldFluenceSensor` | Stokes en campo lejano $(\theta, \phi)$ |
| `FarFieldCBSSensor` | **CBS** en campo lejano — coherente + incoherente |
| `StatisticsSensor` | Histogramas configurables (scatterings, ángulos, tiempos) |

### Coherent Backscattering (CBS)

El fenómeno principal modelado por la librería. Para cada trayectoria forward, se calcula el path reverso usando el **teorema de reciprocidad** ($Q \cdot T^\top \cdot Q$), evitando recalcular cada scattering intermedio. El detector `FarFieldCBSSensor` acumula:

- **Intensidad coherente**: $|E_f + E_r|^2$
- **Intensidad incoherente**: $|E_f|^2 + |E_r|^2$

El *enhancement factor* es $\eta(\theta) = I_\text{coh} / I_\text{incoh}$, con $\eta \to 2$ en backscattering exacto para medios conservadores.

---

## Licencia

[BSD 3-Clause](LICENSE) — Copyright (c) 2025, Nicolas Aguilera
