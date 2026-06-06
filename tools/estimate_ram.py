#!/usr/bin/env python3
"""
estimate_ram.py — Estimador de memoria RAM para simulaciones de luminis-mc (CBS/MC).

Modela cuanta RAM ocupara una corrida de `run_simulation_parallel`, en funcion de:
  - numero de hilos (n_threads)
  - tipo, tamano y resolucion (espacial / angular / temporal) de cada sensor
  - numero de fotones y fraccion detectada (solo para PhotonRecordSensor)
  - grid de absorcion opcional
  - tracking CBS (track_reverse_paths) -> tamano del Photon vivo por hilo

═══════════════════════════════════════════════════════════════════════════════
MODELO DE MEMORIA (derivado del codigo C++ del repo)
═══════════════════════════════════════════════════════════════════════════════

1) PATRON CLONE-MERGE (src/simulation.cpp::run_simulation_parallel)
   Antes de lanzar los hilos se hace `config.detector->clone()` y
   `config.absorption->clone()` UNA VEZ POR HILO. El objeto original (el que tu
   creaste) sigue vivo. => los acumuladores existen (n_threads + 1) veces en
   memoria simultaneamente (N clones de trabajo + 1 master donde se hace merge).

2) Matrix (include/luminis/math/vec.hpp)
   Backing = SmallStorage<double>: array<double,16> inline (128 B) + std::vector
   (24 B) + size_t (8 B) + bool (8 B con padding) + rows/cols uint (8 B)
   => ~176 B de OVERHEAD por objeto Matrix, INDEPENDIENTE del tamano.
   Si rows*cols > 16 => ademas rows*cols*8 B en heap.
   Esto importa cuando hay MUCHAS matrices pequenas (p.ej. N_t alto).

3) CMatrix: array<complex<double>,16> inline (256 B) + 48 B => ~304 B overhead.
   Si rows*cols > 16 => ademas rows*cols*16 B en heap.

4) N_t (bins temporales): si dt>0 => N_t = ceil(len_t/dt) + 1 (bin 0 integrado),
   si dt==0 => N_t = 1. Multiplica TODOS los grids del sensor.

5) PhotonRecordSensor: vector<PhotonRecord>, crece linealmente con los fotones
   DETECTADOS (no con n_photons). PhotonRecord ~= 264 B. Pico durante el merge
   ~= 2x (master con todos + hilos con su parte) => se modela el pico.

6) Photon vivo: 1 por hilo. Con track_reverse_paths lleva 6 Matrix + 4 CMatrix
   (~2.4 KB). Despreciable salvo conteo de hilos enorme, pero se incluye.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

# ─── Constantes de tamano (bytes), derivadas del ABI de C++23 en 64-bit ─────────
DOUBLE = 8
COMPLEX = 16  # std::complex<double>
INT = 4
VEC3 = 3 * DOUBLE          # 24
CVEC2 = 2 * COMPLEX        # 32

# Overhead por OBJETO Matrix / CMatrix (SmallStorage SBO, ver punto 2 y 3).
MATRIX_OVERHEAD = 176
CMATRIX_OVERHEAD = 304

# sizeof(PhotonRecord): uint(8 c/pad) + 6 double(48) + 6 Vec3(144) + 2 CVec2(64)
PHOTON_RECORD = 8 + 6 * DOUBLE + 6 * VEC3 + 2 * CVEC2  # 264

# sizeof(Photon vivo) aproximado, con campos CBS siempre presentes en el struct.
# (los Matrix/CMatrix existen aunque track_reverse_paths este off; off solo evita
#  escribirlos, no los borra del struct) -> se cuenta siempre.
PHOTON_LIVE = (
    6 * MATRIX_OVERHEAD     # P_local, P0, P1, Pn2, Pn1, Pn
    + 4 * CMATRIX_OVERHEAD  # matrix_T (+buffer), matrix_T_raw (+buffer)
    + 200                   # escalares + Vec3 + CVec2 sueltos
)

VECTOR_OVERHEAD = 24  # std::vector de control (3 punteros)


def matrix_bytes(rows: int, cols: int) -> int:
    """Bytes de UN objeto Matrix de rows*cols (overhead + heap si >16 elems)."""
    n = rows * cols
    heap = n * DOUBLE if n > 16 else 0
    return MATRIX_OVERHEAD + heap


def cmatrix_bytes(rows: int, cols: int) -> int:
    n = rows * cols
    heap = n * COMPLEX if n > 16 else 0
    return CMATRIX_OVERHEAD + heap


def n_bins_time(len_t: float, dt: float) -> int:
    """N_t segun la convencion del repo (dt>0 -> ceil(len_t/dt)+1, si no 1)."""
    if dt and dt > 0:
        return int(math.ceil(len_t / dt)) + 1
    return 1


# ═══════════════════════════════════════════════════════════════════════════════
# Definicion de sensores
# ═══════════════════════════════════════════════════════════════════════════════

class Sensor:
    """Base. Cada subclase devuelve los bytes que ocupa UNA instancia."""
    name = "Sensor"

    def bytes_per_instance(self, ctx: "SimSpec") -> int:
        return 0

    def is_per_photon(self) -> bool:
        """True si crece con fotones detectados (no se replica por hilo igual)."""
        return False

    def describe(self, ctx: "SimSpec") -> str:
        return self.name


@dataclass
class FarFieldCBS(Sensor):
    """FarFieldCBSSensor: 8 vector<Matrix> [N_theta x N_phi] x N_t.
    (S0..S3 coherente + S0..S3 incoherente)."""
    theta_max: float
    phi_max: float
    d_theta: float
    d_phi: float
    len_t: float = 0.0
    dt: float = 0.0
    name: str = "FarFieldCBSSensor"

    def _dims(self):
        n_theta = int(math.ceil(self.theta_max / self.d_theta))
        n_phi = int(math.ceil(self.phi_max / self.d_phi))
        n_t = n_bins_time(self.len_t, self.dt)
        return n_theta, n_phi, n_t

    def bytes_per_instance(self, ctx):
        n_theta, n_phi, n_t = self._dims()
        per_matrix = matrix_bytes(n_theta, n_phi)
        # 8 grids, cada uno vector de N_t matrices.
        return 8 * (VECTOR_OVERHEAD + n_t * per_matrix)

    def describe(self, ctx):
        n_theta, n_phi, n_t = self._dims()
        return f"{self.name} [Nθ={n_theta} × Nφ={n_phi} × Nt={n_t}, 8 grids Stokes coh/incoh]"


@dataclass
class PlanarFluence(Sensor):
    """PlanarFluenceSensor: 4 vector<Matrix> [N_x x N_y] x N_t (S0..S3)."""
    len_x: float
    len_y: float
    dx: float
    dy: float
    len_t: float = 0.0
    dt: float = 0.0
    name: str = "PlanarFluenceSensor"

    def _dims(self):
        n_x = int(math.ceil(self.len_x / self.dx))
        n_y = int(math.ceil(self.len_y / self.dy))
        n_t = n_bins_time(self.len_t, self.dt)
        return n_x, n_y, n_t

    def bytes_per_instance(self, ctx):
        n_x, n_y, n_t = self._dims()
        per_matrix = matrix_bytes(n_x, n_y)
        return 4 * (VECTOR_OVERHEAD + n_t * per_matrix)

    def describe(self, ctx):
        n_x, n_y, n_t = self._dims()
        return f"{self.name} [Nx={n_x} × Ny={n_y} × Nt={n_t}, 4 grids Stokes]"


@dataclass
class PlanarField(Sensor):
    """PlanarFieldSensor: Ex, Ey CMatrix [N_x x N_y] (sin resolucion temporal)."""
    len_x: float
    len_y: float
    dx: float
    dy: float
    name: str = "PlanarFieldSensor"

    def _dims(self):
        n_x = int(math.ceil(self.len_x / self.dx))
        n_y = int(math.ceil(self.len_y / self.dy))
        return n_x, n_y

    def bytes_per_instance(self, ctx):
        n_x, n_y = self._dims()
        return 2 * cmatrix_bytes(n_x, n_y)  # Ex, Ey

    def describe(self, ctx):
        n_x, n_y = self._dims()
        return f"{self.name} [Nx={n_x} × Ny={n_y}, Ex/Ey complejos]"


@dataclass
class PhotonRecord(Sensor):
    """PhotonRecordSensor: vector<PhotonRecord>, crece con fotones DETECTADOS."""
    detection_fraction: float = 1.0  # fraccion de n_photons que cruza este sensor
    name: str = "PhotonRecordSensor"

    def is_per_photon(self):
        return True

    def n_records(self, ctx):
        return int(ctx.n_photons * self.detection_fraction)

    def bytes_per_instance(self, ctx):
        # Pico durante merge ~ 2x (master acumulado + hilos con su parte).
        return 2 * self.n_records(ctx) * PHOTON_RECORD

    def describe(self, ctx):
        return (f"{self.name} [~{self.n_records(ctx):,} registros × {PHOTON_RECORD} B, "
                f"pico merge ×2]")


@dataclass
class Statistics(Sensor):
    """StatisticsSensor: histogramas de enteros. Suele ser pequeno.
    Se modela el peor caso con todos los histogramas activos."""
    max_events: int = 0
    n_bins_theta: int = 0
    n_bins_phi: int = 0
    n_bins_depth: int = 0
    n_bins_time: int = 0
    n_bins_weight: int = 0
    n_t: int = 1
    name: str = "StatisticsSensor"

    def bytes_per_instance(self, ctx):
        total = 0
        # events/theta/phi/depth son vector<vector<int>> con N_t filas.
        for n in (self.max_events, self.n_bins_theta, self.n_bins_phi, self.n_bins_depth):
            if n:
                total += VECTOR_OVERHEAD + self.n_t * (VECTOR_OVERHEAD + n * INT)
        for n in (self.n_bins_time, self.n_bins_weight):
            if n:
                total += VECTOR_OVERHEAD + n * INT
        return total

    def describe(self, ctx):
        return f"{self.name} [histogramas, Nt={self.n_t}]"


@dataclass
class AbsorptionGrid:
    """Absorption: vector<Matrix> [n_r x n_z] x n_t (grid cilindrico)."""
    radius: float
    depth: float
    d_r: float
    d_z: float
    len_t: float = 0.0
    dt: float = 0.0

    def _dims(self):
        n_r = int(self.radius / self.d_r) + 1
        n_z = int(self.depth / self.d_z) + 1
        n_t = n_bins_time(self.len_t, self.dt)
        return n_r, n_z, n_t

    def bytes_per_instance(self):
        n_r, n_z, n_t = self._dims()
        return VECTOR_OVERHEAD + n_t * matrix_bytes(n_r, n_z)

    def describe(self):
        n_r, n_z, n_t = self._dims()
        return f"Absorption [n_r={n_r} × n_z={n_z} × Nt={n_t}]"


# ═══════════════════════════════════════════════════════════════════════════════
# Especificacion de la simulacion completa
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimSpec:
    n_photons: int
    n_threads: int
    sensors: list = field(default_factory=list)
    absorption: AbsorptionGrid | None = None
    track_reverse_paths: bool = False


def human(b: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    x = float(b)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    return f"{x:,.2f} {units[i]}"


def estimate(spec: SimSpec, verbose: bool = True) -> dict:
    n_replicas = spec.n_threads + 1  # N clones de trabajo + 1 master
    lines = []
    total_replicated = 0   # acumuladores que se replican (n_threads + 1)
    total_per_photon = 0   # PhotonRecord: pico ya incluido en bytes_per_instance

    for s in spec.sensors:
        per = s.bytes_per_instance(spec)
        if s.is_per_photon():
            # El pico de PhotonRecord (master + hilos) ya esta contemplado dentro
            # de bytes_per_instance (factor 2). No se multiplica por n_replicas.
            sub = per
            total_per_photon += sub
            factor_str = "pico merge (×2)"
        else:
            sub = per * n_replicas
            total_replicated += sub
            factor_str = f"×{n_replicas} (hilos+master)"
        lines.append((s.describe(spec), per, factor_str, sub))

    # Absorcion (tambien clone-merge).
    abs_total = 0
    if spec.absorption:
        per = spec.absorption.bytes_per_instance()
        abs_total = per * n_replicas
        total_replicated += abs_total
        lines.append((spec.absorption.describe(), per,
                      f"×{n_replicas} (hilos+master)", abs_total))

    # Photon vivo: 1 por hilo (el master no corre fotones).
    photon_total = PHOTON_LIVE * spec.n_threads
    lines.append(("Photon vivo (estado de transporte)", PHOTON_LIVE,
                  f"×{spec.n_threads} (1/hilo)", photon_total))

    grand = total_replicated + total_per_photon + photon_total

    if verbose:
        print("═" * 78)
        print(f"  Estimacion de RAM — luminis-mc")
        print(f"  n_photons = {spec.n_photons:,}   n_threads = {spec.n_threads}"
              f"   track_reverse_paths = {spec.track_reverse_paths}")
        print(f"  Replicas de acumuladores = n_threads + 1 = {n_replicas}")
        print("═" * 78)
        for desc, per, factor, sub in lines:
            print(f"  • {desc}")
            print(f"      por instancia: {human(per):>14}   {factor:>22}   "
                  f"=> {human(sub):>14}")
        print("─" * 78)
        print(f"  Acumuladores replicados : {human(total_replicated):>14}")
        print(f"  PhotonRecord (pico)     : {human(total_per_photon):>14}")
        print(f"  Photons vivos           : {human(photon_total):>14}")
        print("─" * 78)
        print(f"  TOTAL ESTIMADO (datos)  : {human(grand):>14}")
        # Margen practico: el RSS real incluye allocator/fragmentacion + runtime.
        print(f"  + ~15% overhead alloc.  : {human(grand * 1.15):>14}  (estimacion RSS realista)")
        print("═" * 78)

    return {
        "replicated_bytes": total_replicated,
        "photon_record_bytes": total_per_photon,
        "photons_live_bytes": photon_total,
        "total_bytes": grand,
        "total_with_overhead": grand * 1.15,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Ejemplo / plantilla. Edita los valores a tu corrida y ejecuta:
#     python tools/estimate_ram.py
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ---- EJEMPLO 1: CBS far-field con resolucion temporal, 14 hilos ----------
    spec = SimSpec(
        n_photons=1_000_000_000,
        n_threads=14,
        track_reverse_paths=True,
        sensors=[
            # CBS angular con tiempo: theta_max, phi_max, d_theta, d_phi, len_t, dt
            FarFieldCBS(
                theta_max=math.radians(2.0),    # cono CBS estrecho
                phi_max=2 * math.pi,
                d_theta=math.radians(2.0) / 200, # 200 bins en theta
                d_phi=2 * math.pi / 360,         # 360 bins en phi
                len_t=5.0, dt=0.05,              # Nt = 101
            ),
            # Histogramas de estadistica (baratos).
            Statistics(max_events=1000, n_bins_theta=200, n_t=101),
        ],
        # absorption=AbsorptionGrid(radius=10, depth=10, d_r=0.05, d_z=0.05,
        #                           len_t=5.0, dt=0.05),
    )
    estimate(spec)

    print()
    # ---- EJEMPLO 2: muestra el efecto del tiempo y los hilos -----------------
    print("Sensibilidad a la resolucion temporal (mismo grid angular, 14 hilos):")
    for dt in (0.0, 0.5, 0.1, 0.02):
        s = SimSpec(
            n_photons=1_000_000_000, n_threads=14, track_reverse_paths=True,
            sensors=[FarFieldCBS(
                theta_max=math.radians(2.0), phi_max=2 * math.pi,
                d_theta=math.radians(2.0) / 200, d_phi=2 * math.pi / 360,
                len_t=5.0, dt=dt)],
        )
        r = estimate(s, verbose=False)
        nt = n_bins_time(5.0, dt)
        print(f"   dt={dt:<5} (Nt={nt:>4}) -> {human(r['total_with_overhead'])}")
