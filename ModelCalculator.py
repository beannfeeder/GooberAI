"""
Trend Model Calculator for YouTube Video Analysis
Implements various trend estimation models and predictions
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class TrendModelCalculator:
    """
    A class to calculate and predict trends in YouTube video data
    """
    
    def __init__(self, data):
        """Initialize with preprocessed data"""
        self.data = data.copy()
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        
    def prepare_features(self):
        """Prepare features for trend modeling"""
        print("Preparing features for trend modeling...")
        
        # Create time-based features
        self.data['days_since_start'] = (self.data['publishedAt'] - self.data['publishedAt'].min()).dt.days
        self.data['month_sin'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['month_cos'] = np.cos(2 * np.pi * self.data['month'] / 12)
        self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        
        # Duration categories as features
        duration_bins = [0, 30, 60, 180, 600, float('inf')]
        duration_labels = ['very_short', 'short', 'medium', 'long', 'very_long']
        self.data['duration_category'] = pd.cut(self.data['duration_seconds'], 
                                               bins=duration_bins, labels=duration_labels)
        
        # Create dummy variables for categorical features
        duration_dummies = pd.get_dummies(self.data['duration_category'], prefix='duration', dummy_na=True)
        self.data = pd.concat([self.data, duration_dummies], axis=1)
        
        # Language encoding (top languages only)
        if not self.data['defaultLanguage'].isna().all():
            top_languages = self.data['defaultLanguage'].value_counts().head(5).index
            for lang in top_languages:
                if pd.notna(lang):  # Check for valid language codes
                    self.data[f'lang_{lang}'] = (self.data['defaultLanguage'] == lang).astype(int)
        
        print("Feature preparation completed.")
        
    def create_trend_features(self):
        """Create trend-specific features"""
        print("Creating trend-specific features...")
        
        # Rolling averages for trend detection
        self.data = self.data.sort_values('publishedAt')
        
        # Calculate rolling statistics by channel
        channel_stats = []
        unique_channels = self.data['channelId'].unique()
        print(f"Processing {len(unique_channels)} unique channels...")
        
        for i, channel_id in enumerate(unique_channels):
            if i % 10000 == 0:  # Progress indicator
                print(f"Processed {i}/{len(unique_channels)} channels...")
                
            channel_data = self.data[self.data['channelId'] == channel_id].copy()
            if len(channel_data) > 1:
                channel_data = channel_data.reset_index(drop=True)
                channel_data['channel_avg_views'] = channel_data['viewCount'].expanding().mean()
                channel_data['channel_avg_engagement'] = channel_data['engagement_score'].expanding().mean()
                channel_data['video_number'] = range(1, len(channel_data) + 1)
            else:
                channel_data['channel_avg_views'] = channel_data['viewCount']
                channel_data['channel_avg_engagement'] = channel_data['engagement_score']
                channel_data['video_number'] = 1
            channel_stats.append(channel_data)
        
        self.data = pd.concat(channel_stats, ignore_index=True)
        
        # Global trend features
        self.data['global_trend_views'] = self.data.groupby('days_since_start')['viewCount'].transform('mean')
        self.data['global_trend_engagement'] = self.data.groupby('days_since_start')['engagement_score'].transform('mean')
        
        print("Trend features created.")
        
    def build_view_prediction_model(self):
        """Build a model to predict view counts"""
        print("Building view prediction model...")
        
        # Select features for modeling
        feature_cols = [
            'days_since_start', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
            'duration_seconds', 'video_number', 'channel_avg_views', 'channel_avg_engagement'
        ]
        
        # Add duration dummy variables
        duration_cols = [col for col in self.data.columns if col.startswith('duration_')]
        feature_cols.extend(duration_cols)
        
        # Add language dummy variables
        lang_cols = [col for col in self.data.columns if col.startswith('lang_')]
        feature_cols.extend(lang_cols)
        
        # Prepare data
        X = self.data[feature_cols].fillna(0)
        y = np.log1p(self.data['viewCount'])  # Log transform to handle skewness
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"{name}: MSE={mse:.4f}, R²={r2:.4f}")
        
        self.models['view_prediction'] = results
        self.scalers['view_prediction'] = scaler
        
        # Store test data for visualization
        self.test_data = {
            'X_test': X_test,
            'y_test': y_test,
            'y_test_original': np.expm1(y_test)
        }
        
        return results