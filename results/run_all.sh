#!/bin/bash

# Carpeta donde están tus scripts
cd /home/niaggar/Developer/luminis-mc

# Activa el entorno virtual (ajusta la ruta si tu .venv está en otro lado)
source .venv/bin/activate

# Crea la carpeta de logs si no existe
mkdir -p logs

# Lista de scripts en el orden que quieres que corran
scripts=(
    "results.study_mixture_layer_1.study_mixture_layer_MU_S_TOTAL_FIXED__PCIR"
    "results.study_mixture_layer_1.study_mixture_layer_MU_S_TOTAL_FIXED__PLIN"
)

echo "=== Inicio del batch: $(date) ===" | tee -a logs/batch_summary.log
echo "=== Python usado: $(which python) ===" | tee -a logs/batch_summary.log

for script in "${scripts[@]}"; do
    nombre=$(basename "$script" .py)
    log_file="logs/${nombre}.log"

    echo "=== Iniciando $script ($(date)) ===" | tee -a logs/batch_summary.log

    python -m "$script" > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "=== $script terminó OK ($(date)) ===" | tee -a logs/batch_summary.log
    else
        echo "=== $script FALLÓ ($(date)) — revisa $log_file ===" | tee -a logs/batch_summary.log
    fi
done

echo "=== Batch completo: $(date) ===" | tee -a logs/batch_summary.log

deactivate