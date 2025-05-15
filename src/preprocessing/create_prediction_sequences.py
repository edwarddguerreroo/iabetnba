import pandas as pd
import numpy as np
import os
import argparse
import json
import logging
from tqdm import tqdm
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional

from src.preprocessing.data_loader import NBADataLoader
from src.preprocessing.feature_engineering import FeatureEngineering
from src.preprocessing.sequences import SequenceGenerator, save_sequences, BETTING_LINES, load_sequences

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction_sequences.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('PredictionSequences')

# Modelos de predicción y sus targets correspondientes
PREDICTION_MODELS = {
    # Modelos de equipo
    'win_predictor': {
        'target': 'Win',
        'lines': BETTING_LINES['Win'],
        'sequence_length': 10,
        'description': 'Predicción de victoria/derrota de equipo'
    },
    'total_points_predictor': {
        'target': 'Total_Points_Over_Under',
        'lines': BETTING_LINES['Total_Points_Over_Under'],
        'sequence_length': 10,
        'description': 'Predicción de over/under de puntos totales en un partido'
    },
    'team_points_predictor': {
        'target': 'Team_Points_Over_Under',
        'lines': BETTING_LINES['Team_Points_Over_Under'],
        'sequence_length': 10,
        'description': 'Predicción de over/under de puntos de un equipo'
    },
    
    # Modelos de jugador
    'pts_predictor': {
        'target': 'PTS',
        'lines': BETTING_LINES['PTS'],
        'sequence_length': 10,
        'description': 'Predicción de over/under de puntos de un jugador'
    },
    'trb_predictor': {
        'target': 'TRB',
        'lines': BETTING_LINES['TRB'],
        'sequence_length': 10,
        'description': 'Predicción de over/under de rebotes de un jugador'
    },
    'ast_predictor': {
        'target': 'AST',
        'lines': BETTING_LINES['AST'],
        'sequence_length': 10,
        'description': 'Predicción de over/under de asistencias de un jugador'
    },
    '3p_predictor': {
        'target': '3P',
        'lines': BETTING_LINES['3P'],
        'sequence_length': 10,
        'description': 'Predicción de over/under de triples de un jugador'
    },
    'double_double_predictor': {
        'target': 'Double_Double',
        'lines': BETTING_LINES['Double_Double'],
        'sequence_length': 10,
        'description': 'Predicción de doble-doble de un jugador'
    },
    'triple_double_predictor': {
        'target': 'Triple_Double',
        'lines': BETTING_LINES['Triple_Double'],
        'sequence_length': 10,
        'description': 'Predicción de triple-doble de un jugador'
    }
}

def load_and_preprocess_data(game_data_path: str, biometrics_path: str, output_dir: str) -> pd.DataFrame:
    """
    Carga y preprocesa los datos utilizando el DataLoader y FeatureEngineering
    
    Args:
        game_data_path: Ruta al archivo CSV con datos de partidos
        biometrics_path: Ruta al archivo CSV con datos biométricos
        output_dir: Directorio donde guardar el archivo CSV de características
        
    Returns:
        DataFrame con todos los datos procesados y características generadas
    """
    logger.info("Cargando datos...")
    data_loader = NBADataLoader(game_data_path, biometrics_path)
    df = data_loader.load_data()
    
    # Asegurarse de que Date sea datetime
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            logger.info("Columna Date convertida a datetime correctamente")
        except Exception as e:
            logger.error(f"Error al convertir Date a datetime: {str(e)}")
    else:
        logger.warning("Columna Date no encontrada en los datos")
    
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    logger.info("Generando características...")
    feature_eng = FeatureEngineering(
        window_sizes=[3, 10, 20],
        correlation_threshold=0.95,
        enable_correlation_analysis=True,
        n_jobs=4  # Usar 4 procesos en paralelo
    )
    
    df_processed = feature_eng.generate_all_features(df)
    logger.info(f"Características generadas: {df_processed.shape[1] - df.shape[1]} nuevas columnas")
    
    # Asegurar que el directorio de salida está definido
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Guardar DataFrame con características generadas en un archivo CSV
    features_csv_path = os.path.join(output_dir, 'features_generated.csv')
    df_processed.to_csv(features_csv_path, index=False)
    logger.info(f"Características generadas guardadas en {features_csv_path}")
    
    return df_processed

def create_sequences_for_model(
    df: pd.DataFrame, 
    model_name: str, 
    output_dir: str,
    min_games: int = 5,
    confidence_threshold: float = 0.7,
    min_historical_accuracy: float = 0.8
) -> Dict:
    """
    Crea secuencias para un modelo específico
    
    Args:
        df: DataFrame con datos procesados
        model_name: Nombre del modelo para el que generar secuencias
        output_dir: Directorio donde guardar las secuencias
        min_games: Número mínimo de juegos para procesar un jugador
        confidence_threshold: Umbral de confianza para predicciones
        min_historical_accuracy: Precisión histórica mínima requerida
        
    Returns:
        Diccionario con información sobre las secuencias generadas
    """
    if model_name not in PREDICTION_MODELS:
        logger.error(f"Modelo no reconocido: {model_name}")
        return {}
    
    model_config = PREDICTION_MODELS[model_name]
    target = model_config['target']
    lines = model_config['lines']
    sequence_length = model_config['sequence_length']
    
    logger.info(f"Generando secuencias para {model_name} (target: {target})")
    logger.info(f"Líneas de apuesta: {lines}")
    
    # Crear generador de secuencias específico para este modelo
    sequence_generator = SequenceGenerator(
        sequence_length=sequence_length,
        target_columns=[target],
        model_type=model_name,
        confidence_threshold=confidence_threshold,
        min_historical_accuracy=min_historical_accuracy,
        min_samples=min_games
    )
    
    # Generar secuencias
    sequences, targets, categorical, line_values, betting_insights = sequence_generator.generate_sequences(
        df=df,
        min_games=min_games,
        null_threshold=0.9,
        use_target_specific_features=True
    )
    
    # Verificar si se generaron secuencias
    if len(sequences) == 0:
        logger.warning(f"No se generaron secuencias para {model_name}")
        return {
            'model_name': model_name,
            'target': target,
            'sequences_generated': 0,
            'status': 'ERROR: No se generaron secuencias'
        }
    
    # Guardar secuencias
    output_path = os.path.join(output_dir, f"{model_name}_sequences.npz")
    save_sequences(
        sequences=sequences,
        targets=targets,
        categorical=categorical,
        line_values=line_values,
        output_path=output_path
    )
    
    # Guardar insights de apuestas
    insights_path = os.path.join(output_dir, f"{model_name}_insights.json")
    with open(insights_path, 'w') as f:
        json.dump(betting_insights, f, indent=2)
    
    return {
        'model_name': model_name,
        'target': target,
        'sequences_generated': len(sequences),
        'shape': sequences.shape,
        'output_path': output_path,
        'insights_path': insights_path,
        'status': 'SUCCESS'
    }

def create_lightgbm_model(
    sequences: np.ndarray,
    targets: np.ndarray,
    categorical: np.ndarray,
    line_values: np.ndarray,
    output_dir: str,
    model_name: str,
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Dict:
    """
    Crea y entrena un modelo LightGBM para las secuencias generadas
    
    Args:
        sequences: Array de secuencias
        targets: Array de targets
        categorical: Array de variables categóricas
        line_values: Array de valores de línea
        output_dir: Directorio donde guardar el modelo
        model_name: Nombre del modelo
        test_size: Proporción del conjunto de prueba
        val_size: Proporción del conjunto de validación
        
    Returns:
        Diccionario con información sobre el modelo entrenado
    """
    from sklearn.model_selection import train_test_split
    
    logger.info(f"Preparando datos para modelo LightGBM: {model_name}")
    
    # Aplanar las secuencias para LightGBM (no usa directamente secuencias temporales)
    # Convertir [n_samples, seq_len, n_features] a [n_samples, seq_len * n_features]
    n_samples, seq_len, n_features = sequences.shape
    X = sequences.reshape(n_samples, seq_len * n_features)
    
    # Añadir variables categóricas si están disponibles
    if categorical is not None and categorical.size > 0:
        X = np.hstack([X, categorical])
    
    # Añadir valores de línea si están disponibles
    if line_values is not None and line_values.size > 0:
        X = np.hstack([X, line_values])
    
    # Dividir en conjuntos de entrenamiento, validación y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, targets, test_size=test_size, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size/(1-test_size), random_state=42
    )
    
    # Crear conjuntos de datos para LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Parámetros para LightGBM (optimizados para clasificación binaria)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Entrenar modelo
    logger.info(f"Entrenando modelo LightGBM para {model_name}...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Evaluar en conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    
    y_pred_binary = (y_pred > 0.5).astype(int)
    metrics = {
        'auc': roc_auc_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'precision': precision_score(y_test, y_pred_binary),
        'recall': recall_score(y_test, y_pred_binary),
        'f1': f1_score(y_test, y_pred_binary)
    }
    
    # Guardar modelo
    model_path = os.path.join(output_dir, f"{model_name}_model.txt")
    model.save_model(model_path)
    
    # Guardar métricas
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return {
        'model_name': model_name,
        'metrics': metrics,
        'model_path': model_path,
        'metrics_path': metrics_path,
        'best_iteration': model.best_iteration,
        'feature_importance': model.feature_importance().tolist()
    }

def main(
    game_data_path: str,
    biometrics_path: str,
    output_dir: str,
    models: List[str] = None,
    train_models: bool = True,
    model_group: str = 'all'  # Nuevo parámetro: 'team', 'player', 'all'
):
    """
    Función principal para generar secuencias y entrenar modelos
    
    Args:
        game_data_path: Ruta al archivo CSV con datos de partidos
        biometrics_path: Ruta al archivo CSV con datos biométricos
        output_dir: Directorio donde guardar las secuencias y modelos
        models: Lista de modelos a generar (None = todos)
        train_models: Si entrenar modelos después de generar secuencias
        model_group: Tipo de modelos a generar: 'team', 'player', 'all'
    """
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar y preprocesar datos
    df = load_and_preprocess_data(game_data_path, biometrics_path, output_dir)
    
    # Verificar que tenemos datos
    if df.empty:
        logger.error("No hay datos para procesar")
        return
    
    # Determinar qué modelos generar
    all_models = list(PREDICTION_MODELS.keys())
    team_models = ['win_predictor', 'total_points_predictor', 'team_points_predictor']
    player_models = [m for m in all_models if m not in team_models]
    
    if model_group == 'team':
        available_models = team_models
    elif model_group == 'player':
        available_models = player_models
    else:
        available_models = all_models
        
    if models is None:
        models = available_models
    else:
        # Filtrar solo modelos válidos y disponibles según el grupo seleccionado
        models = [m for m in models if m in available_models]
    
    logger.info(f"Generando secuencias para {len(models)} modelos: {', '.join(models)}")
    
    # Asegurar que tenemos columnas necesarias para modelos de equipo
    if any(model in team_models for model in models):
        # Verificar y crear columnas necesarias para modelos de equipo
        if 'Total_Points' not in df.columns and 'Team_Points' in df.columns and 'Opp_Points' in df.columns:
            df['Total_Points'] = df['Team_Points'] + df['Opp_Points']
            logger.info("Columna Total_Points creada a partir de Team_Points y Opp_Points")
    
    # Generar secuencias para cada modelo
    results = []
    for model_name in models:
        logger.info(f"Procesando modelo: {model_name}")
        
        # Determinar si es modelo de equipo o jugador
        is_team_model = model_name in team_models
        
        # Verificar columnas requeridas para este modelo
        target = PREDICTION_MODELS[model_name]['target']
        if target not in df.columns and model_name in team_models:
            # Para modelos de equipo, intentar crear la columna target si es posible
            if target == 'Total_Points_Over_Under' and 'Total_Points' in df.columns:
                # No necesitamos hacer nada, usaremos Total_Points directamente
                pass
            elif target == 'Team_Points_Over_Under' and 'Team_Points' in df.columns:
                # No necesitamos hacer nada, usaremos Team_Points directamente
                pass
            else:
                logger.warning(f"No se puede generar el modelo {model_name}, falta columna: {target}")
                results.append({
                    'model_name': model_name,
                    'target': target,
                    'sequences_generated': 0,
                    'status': f'ERROR: Falta columna {target}'
                })
                continue
        
        # Generar secuencias
        result = create_sequences_for_model(
            df=df,
            model_name=model_name,
            output_dir=output_dir,
            min_games=5,
            confidence_threshold=0.7,
            min_historical_accuracy=0.8
        )
        
        results.append(result)
        
        # Si se generaron secuencias y se solicitó entrenar modelos, hacerlo
        if (train_models and 
            result.get('status', '').startswith('SUCCESS') and
            'output_path' in result):
            
            # Cargar secuencias
            data = load_sequences(result['output_path'])
            
            # Entrenar modelo
            model_result = create_lightgbm_model(
                sequences=data['sequences'],
                targets=data['targets'],
                categorical=data['categorical'],
                line_values=data.get('line_values'),
                output_dir=output_dir,
                model_name=model_name
            )
            
            # Actualizar resultado con información del modelo
            result.update(model_result)
    
    # Guardar resumen
    summary_path = os.path.join(output_dir, 'sequence_generation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Mostrar resumen
    success_count = sum(1 for r in results if r.get('status', '').startswith('SUCCESS'))
    error_count = len(results) - success_count
    
    logger.info(f"Proceso completado. Resumen guardado en {summary_path}")
    logger.info(f"Resumen: {success_count} modelos generados correctamente, {error_count} con errores")
    
    # Usar símbolos ASCII en lugar de Unicode para evitar problemas de codificación
    for result in results:
        model_name = result.get('model_name', '')
        if result.get('status', '').startswith('SUCCESS'):
            sequences_count = result.get('sequences_generated', 0)
            logger.info(f"+ {model_name}: {sequences_count} secuencias generadas")
        else:
            error_msg = result.get('status', 'ERROR desconocido')
            logger.error(f"- {model_name}: {error_msg}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generación de secuencias para modelos de predicción")
    
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
    
    main(
        game_data_path=args.game_data,
        biometrics_path=args.biometrics,
        output_dir=args.output_dir,
        models=args.models,
        train_models=not args.no_train,
        model_group=args.model_group
    ) 