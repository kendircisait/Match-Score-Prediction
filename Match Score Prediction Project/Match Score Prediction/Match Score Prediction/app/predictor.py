import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

class MatchPredictor:
    def __init__(self):
        self.home_model = None
        self.away_model = None
        self.scaler = StandardScaler()
    
    def load_models(self, model_dir='model'):
        """Kaydedilmiş modelleri ve scaler'ı yükle"""
        try:
            self.home_model = joblib.load(os.path.join(model_dir, 'home_goals_model.joblib'))
            self.away_model = joblib.load(os.path.join(model_dir, 'away_goals_model.joblib'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
            return True
        except FileNotFoundError:
            print("Kaydedilmiş model dosyaları bulunamadı.")
            return False
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            return False
    
    def predict_match(self, home_team, away_team, matches_df, stats_df):
        """Maç sonucunu tahmin et"""
        try:
            # Son 5 maçı bul
            home_last_5 = matches_df[
                ((matches_df['home_team'] == home_team) | (matches_df['away_team'] == home_team))
            ].tail(5)
            
            away_last_5 = matches_df[
                ((matches_df['home_team'] == away_team) | (matches_df['away_team'] == away_team))
            ].tail(5)
            
            # Son 5 maç istatistiklerini hesapla
            home_last_5_goals_scored = 0
            home_last_5_goals_conceded = 0
            away_last_5_goals_scored = 0
            away_last_5_goals_conceded = 0
            
            for _, match in home_last_5.iterrows():
                if match['home_team'] == home_team:
                    home_last_5_goals_scored += match['home_goals']
                    home_last_5_goals_conceded += match['away_goals']
                else:
                    home_last_5_goals_scored += match['away_goals']
                    home_last_5_goals_conceded += match['home_goals']
            
            for _, match in away_last_5.iterrows():
                if match['home_team'] == away_team:
                    away_last_5_goals_scored += match['home_goals']
                    away_last_5_goals_conceded += match['away_goals']
                else:
                    away_last_5_goals_scored += match['away_goals']
                    away_last_5_goals_conceded += match['home_goals']
            
            # Takım istatistiklerini al
            home_stats = stats_df[stats_df['team_name'] == home_team].iloc[0]
            away_stats = stats_df[stats_df['team_name'] == away_team].iloc[0]
            
            # Feature vektörünü oluştur
            features = [
                home_stats['G'] / home_stats['OM'],
                home_stats['B'] / home_stats['OM'],
                home_stats['M'] / home_stats['OM'],
                home_stats['AG'] / home_stats['OM'],
                home_stats['YG'] / home_stats['OM'],
                home_stats['P'] / home_stats['OM'],
                home_stats['A'],
                home_last_5_goals_scored / max(len(home_last_5), 1),
                home_last_5_goals_conceded / max(len(home_last_5), 1),
                
                away_stats['G'] / away_stats['OM'],
                away_stats['B'] / away_stats['OM'],
                away_stats['M'] / away_stats['OM'],
                away_stats['AG'] / away_stats['OM'],
                away_stats['YG'] / away_stats['OM'],
                away_stats['P'] / away_stats['OM'],
                away_stats['A'],
                away_last_5_goals_scored / max(len(away_last_5), 1),
                away_last_5_goals_conceded / max(len(away_last_5), 1),
                
                (home_stats['P'] - away_stats['P']) / home_stats['OM'],
                home_stats['A'] - away_stats['A']
            ]
            
            # Feature'ları standardize et
            features_scaled = self.scaler.transform([features])
            
            # Tahminleri yap
            predicted_home_goals = max(0, round(self.home_model.predict(features_scaled)[0]))
            predicted_away_goals = max(0, round(self.away_model.predict(features_scaled)[0]))
            
            return predicted_home_goals, predicted_away_goals
            
        except Exception as e:
            print(f"Tahmin hatası: {str(e)}")
            raise e