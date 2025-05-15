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
    parser.add_argument("--models", nargs="+", 
                        help="Lista de modelos a procesar (por defecto: todos)")
    parser.add_argument("--no_train", action="store_true", 
                        help="No entrenar modelos, solo generar secuencias")
    
    # Modelos específicos disponibles
    available_models = [
        'win_predictor',
        'total_points_predictor',
        'team_points_predictor',
        'pts_predictor',
        'trb_predictor',
        'ast_predictor',
        '3p_predictor',
        'double_double_predictor',
        'triple_double_predictor'
    ]
    
    # Grupos de modelos predefinidos
    model_groups = {
        'team': ['win_predictor', 'total_points_predictor', 'team_points_predictor'],
        'player': ['pts_predictor', 'trb_predictor', 'ast_predictor', '3p_predictor', 
                  'double_double_predictor', 'triple_double_predictor'],
        'all': available_models
    }
    
    parser.add_argument("--model_group", choices=list(model_groups.keys()),
                        help=f"Grupo de modelos predefinidos a ejecutar: {list(model_groups.keys())}")
    
    args = parser.parse_args()
    
    # Determinar qué modelos procesar
    models_to_process = None
    if args.model_group:
        models_to_process = model_groups[args.model_group]
        print(f"Usando grupo de modelos '{args.model_group}': {models_to_process}")
    elif args.models:
        models_to_process = args.models
        print(f"Usando modelos especificados: {models_to_process}")
    
    # Verificar que los archivos de datos existen
    if not os.path.exists(args.game_data):
        print(f"ERROR: El archivo de datos de partidos no existe: {args.game_data}")
        return 1
    
    if not os.path.exists(args.biometrics):
        print(f"ERROR: El archivo de datos biométricos no existe: {args.biometrics}")
        return 1
    
    # Ejecutar la generación de secuencias
    print(f"Iniciando generación de secuencias...")
    print(f"- Datos de partidos: {args.game_data}")
    print(f"- Datos biométricos: {args.biometrics}")
    print(f"- Directorio de salida: {args.output_dir}")
    print(f"- Entrenar modelos: {not args.no_train}")
    
    try:
        run_sequence_generation(
            game_data_path=args.game_data,
            biometrics_path=args.biometrics,
            output_dir=args.output_dir,
            models=models_to_process,
            train_models=not args.no_train
        )
        print("Generación de secuencias completada con éxito")
        return 0
    except Exception as e:
        print(f"ERROR: La generación de secuencias falló: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 