import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import json
from dateutil.relativedelta import relativedelta
import re
import time
import warnings
from src.preprocessing.result_parser import ResultParser


class NBADataLoader:
    """
    Cargador de datos para estadísticas NBA y datos biométricos
    """
    def __init__(self, game_data_path, biometrics_path):
        """
        Inicializa el cargador de datos
        
        Args:
            game_data_path (str): Ruta al archivo CSV con datos de partidos
            biometrics_path (str): Ruta al archivo CSV con datos biométricos
        """
        self.game_data_path = game_data_path
        self.biometrics_path = biometrics_path
        self.result_parser = ResultParser()
        
    def load_data(self):
        """
        Carga, valida y combina los datos de partidos y biométricos
        
        Returns:
            pd.DataFrame: DataFrame combinado con todos los datos procesados
        """
        # Cargar datos
        game_data = pd.read_csv(self.game_data_path)
        biometrics = pd.read_csv(self.biometrics_path)
        
        # Validar datos
        self._validate_game_data(game_data)
        self._validate_biometrics(biometrics)
        
        # Procesar datos
        game_data = self._preprocess_game_data(game_data)
        biometrics = self._preprocess_biometrics(biometrics)
        
        # Combinar datasets
        merged_data = self._merge_datasets(game_data, biometrics)
        
        return merged_data
    
    def _validate_game_data(self, df):
        """Valida el DataFrame de datos de partidos"""
        required_columns = [
            'Player', 'Date', 'Team', 'Opp', 'Result', 'MP',
            'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en datos de partidos: {missing_cols}")
    
    def _validate_biometrics(self, df):
        """Valida el DataFrame de datos biométricos"""
        required_columns = ['Player', 'Height', 'Weight']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en datos biométricos: {missing_cols}")
    
    def _preprocess_game_data(self, df):
        """
        Preprocesa los datos de partidos
        
        - Convierte fechas
        - Parsea resultados
        - Calcula is_home basado en Away
        - Limpia y valida valores
        """
        # Copiar DataFrame
        df = df.copy()
        
        # Convertir fechas
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Parsear resultados
        df = self.result_parser.parse_dataframe(df)
        
        # Calcular is_home basado en Away
        # Away es '@' cuando es visitante, '' cuando es local
        df['is_home'] = (df['Away'] != '@').astype(int)
        
        # Limpiar valores no válidos
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        # Convertir porcentajes a decimales si es necesario
        pct_columns = [col for col in df.columns if col.endswith('%')]
        for col in pct_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100
        
        # Ordenar por jugador y fecha
        df = df.sort_values(['Player', 'Date'])
        
        return df
    
    def _preprocess_biometrics(self, df):
        """
        Preprocesa los datos biométricos
        
        - Convierte altura a pulgadas
        - Limpia y valida valores
        """
        df = df.copy()
        
        # Convertir altura a pulgadas
        def height_to_inches(height_str):
            try:
                feet, inches = height_str.replace('"', '').split("'")
                return int(feet) * 12 + int(inches)
            except:
                return np.nan
        
        df['Height_Inches'] = df['Height'].apply(height_to_inches)
        
        # Validar peso
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
        
        return df
    
    def _merge_datasets(self, game_data, biometrics):
        """
        Combina los datos de partidos con los datos biométricos
        """
        # Merge en base al nombre del jugador
        merged = pd.merge(
            game_data,
            biometrics[['Player', 'Height_Inches', 'Weight']],
            on='Player',
            how='left'
        )
        
        # Verificar que no perdimos datos
        if len(merged) != len(game_data):
            print("¡Advertencia! Algunos jugadores no tienen datos biométricos")
            
        # Calcular métricas adicionales
        merged['BMI'] = (merged['Weight'] * 703) / (merged['Height_Inches'] ** 2)
        
        return merged
    
    
