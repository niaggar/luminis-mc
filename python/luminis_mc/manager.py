import os
import json
import shutil
import inspect
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

class Experiment:
    def __init__(self, name: str, base_dir: str = "sim_results"):
        # Crea una carpeta única: sim_results/2026-02-17_mie_test
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = Path(base_dir) / f"{timestamp}_{name}"
        self.path.mkdir(parents=True, exist_ok=True)
        self.params = {}
        print(f"🧪 Experimento iniciado en: {self.path}")

    def log_params(self, **kwargs):
        """Guarda variables simples (float, string, int) en el registro json."""
        self.params.update(kwargs)
        # Actualizamos el archivo en tiempo real por si crashea la simulación
        self._flush_params()

    def log_script(self, file_path):
        """Guarda una copia del script que ejecutó la simulación (REPRODUCIBILIDAD)"""
        if os.path.exists(file_path):
            shutil.copy(file_path, self.path / "script_snapshot.py")

    def save_sensor(self, sensor, name: str):
        """
        Extrae automáticamente los campos de datos de un sensor C++ y los guarda.
        Intenta detectar si son matrices (S0, S1, Ex, Ey) y los guarda como .npy
        """
        sensor_dir = self.path / name
        sensor_dir.mkdir(exist_ok=True)
        
        # Introspección básica: buscamos atributos que parezcan datos
        # Esto depende de qué expusiste en pybind11. 
        # Asumo que expusiste vectores/matrices como properties.
        
        attributes = dir(sensor)
        saved_keys = []
        
        for attr in attributes:
            if attr.startswith("__"): continue
            val = getattr(sensor, attr)
            
            # Si es lista o vector de C++ convertible a numpy
            if isinstance(val, (list, np.ndarray)) or "vector" in str(type(val)):
                try:
                    arr = np.array(val, copy=False)
                    # Solo guardamos si tiene tamaño considerable
                    if arr.size > 1:
                        np.save(sensor_dir / f"{attr}.npy", arr)
                        saved_keys.append(attr)
                except:
                    pass
        
        print(f"   💾 Sensor '{name}' guardado: {saved_keys}")

    def save_figure(self, fig, name: str):
        """Guarda una figura de matplotlib"""
        fig.savefig(self.path / f"{name}.png", dpi=150)
        fig.savefig(self.path / f"{name}.pdf") # Vectorial también
        
    def _flush_params(self):
        # Convertimos tipos de numpy a nativos para que JSON no se queje
        clean_params = {}
        for k, v in self.params.items():
            if isinstance(v, (np.integer, int)): clean_params[k] = int(v)
            elif isinstance(v, (np.floating, float)): clean_params[k] = float(v)
            elif isinstance(v, (np.ndarray, list)): clean_params[k] = str(v) # O lista si prefieres
            else: clean_params[k] = str(v)
            
        with open(self.path / "parameters.json", 'w') as f:
            json.dump(clean_params, f, indent=4)

# --- Clase para Cargar Datos después ---
class ResultsLoader:
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No existe {path}")
            
        # Cargar parámetros
        with open(self.path / "parameters.json", 'r') as f:
            self.params = json.load(f)
            
    def get_sensor(self, name: str, attr: str):
        """Carga perezosa de arrays numpy: loader.get_sensor('fluence', 'S0_t')"""
        file = self.path / name / f"{attr}.npy"
        return np.load(file)
    
    def __repr__(self):
        return f"ResultsLoader({self.path.name})"