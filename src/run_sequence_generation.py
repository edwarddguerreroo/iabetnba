#!/usr/bin/env python
"""
Script para ejecutar la generación de secuencias para modelos de predicción NBA
"""
import os
import argparse
import sys
from src.preprocessing.create_prediction_sequences import main as run_sequence_generation

def main():
    """Función principal para ejecutar la generación de secuencias"""
    parser = argparse.ArgumentParser(description="Generación de secuencias para modelos de predicción NBA")
    
    # Argumentos requeridos
    parser.add_argument("--game_data", required=True, 
                        help="Ruta al archivo CSV con datos de partidos")
    parser.add_argument("--biometrics", required=True, 
                        help="Ruta al archivo CSV con datos biométricos")
    
    # Argumentos opcionales
    parser.add_argument("--output_dir", default="./model_data", 
                        help="Directorio donde guardar las secuencias y modelos")
    parser.add_argument("--models", nargs='+', default=None,
                        help="Lista de modelos a generar (por defecto: todos)")
    parser.add_argument("--no_train", action="store_true", 
                        help="No entrenar modelos después de generar secuencias")
    parser.add_argument("--model_group", choices=['team', 'player', 'all'], default='all',
                        help="Tipo de modelos a generar: 'team', 'player', 'all'")
    
    args = parser.parse_args()
    
    # Verificar que los archivos existen
    if not os.path.exists(args.game_data):
        print(f"ERROR: Archivo de datos de partidos no encontrado: {args.game_data}")
        return 1
    
    if not os.path.exists(args.biometrics):
        print(f"ERROR: Archivo de datos biométricos no encontrado: {args.biometrics}")
        return 1
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ejecutar la generación de secuencias
    print(f"Iniciando generación de secuencias para modelos de tipo: {args.model_group}")
    print(f"Datos de partidos: {args.game_data}")
    print(f"Datos biométricos: {args.biometrics}")
    print(f"Directorio de salida: {args.output_dir}")
    
    if args.models:
        print(f"Modelos seleccionados: {', '.join(args.models)}")
    else:
        print("Generando todos los modelos disponibles del grupo seleccionado")
    
    if args.no_train:
        print("Solo generando secuencias (sin entrenamiento de modelos)")
    
    # Llamar a la función principal
    results = run_sequence_generation(
        game_data_path=args.game_data,
        biometrics_path=args.biometrics,
        output_dir=args.output_dir,
        models=args.models,
        train_models=not args.no_train,
        model_group=args.model_group
    )
    
    # Verificar resultados
    if not results:
        print("ERROR: No se obtuvieron resultados de la generación de secuencias")
        return 1
    
    # Contar éxitos y errores
    success_count = sum(1 for r in results if r.get('status', '').startswith('SUCCESS'))
    error_count = len(results) - success_count
    
    print(f"\nResumen final:")
    print(f"- Modelos generados correctamente: {success_count}")
    print(f"- Modelos con errores: {error_count}")
    
    # Mostrar detalles de los modelos generados
    if success_count > 0:
        print("\nModelos generados correctamente:")
        for result in results:
            if result.get('status', '').startswith('SUCCESS'):
                model_name = result.get('model_name', '')
                sequences_count = result.get('sequences_generated', 0)
                print(f"  + {model_name}: {sequences_count} secuencias")
    
    # Mostrar detalles de los errores
    if error_count > 0:
        print("\nModelos con errores:")
        for result in results:
            if not result.get('status', '').startswith('SUCCESS'):
                model_name = result.get('model_name', '')
                error_msg = result.get('status', 'ERROR desconocido')
                print(f"  - {model_name}: {error_msg}")
    
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 