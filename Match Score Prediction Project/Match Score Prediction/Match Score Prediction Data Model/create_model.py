import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import joblib

class MatchPredictor:
    def __init__(self):
        self.home_model = None
        self.away_model = None
        self.scaler = StandardScaler()
        
    def create_feature_dataset(self, matches_df, stats_df):
        features = []
        home_goals = []
        away_goals = []
        
        for _, match in matches_df.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            match_no = match['match_no']
            
            # Son 5 maçın performansını hesapla
            home_last_5 = matches_df[
                ((matches_df['home_team'] == home_team) | (matches_df['away_team'] == home_team)) &
                (matches_df['match_no'] < match_no)
            ].tail(5)
            
            away_last_5 = matches_df[
                ((matches_df['home_team'] == away_team) | (matches_df['away_team'] == away_team)) &
                (matches_df['match_no'] < match_no)
            ].tail(5)
            
            # Son 5 maç istatistikleri
            home_last_5_goals_scored = 0
            home_last_5_goals_conceded = 0
            away_last_5_goals_scored = 0
            away_last_5_goals_conceded = 0
            
            for _, last_match in home_last_5.iterrows():
                if last_match['home_team'] == home_team:
                    home_last_5_goals_scored += last_match['home_goals']
                    home_last_5_goals_conceded += last_match['away_goals']
                else:
                    home_last_5_goals_scored += last_match['away_goals']
                    home_last_5_goals_conceded += last_match['home_goals']
                    
            for _, last_match in away_last_5.iterrows():
                if last_match['home_team'] == away_team:
                    away_last_5_goals_scored += last_match['home_goals']
                    away_last_5_goals_conceded += last_match['away_goals']
                else:
                    away_last_5_goals_scored += last_match['away_goals']
                    away_last_5_goals_conceded += last_match['home_goals']
            
            home_stats = stats_df[stats_df['team_name'] == home_team].iloc[0]
            away_stats = stats_df[stats_df['team_name'] == away_team].iloc[0]
            
            match_features = [
                home_stats['G'] / home_stats['OM'],  # Galibiyet oranı
                home_stats['B'] / home_stats['OM'],  # Beraberlik oranı
                home_stats['M'] / home_stats['OM'],  # Mağlubiyet oranı
                home_stats['AG'] / home_stats['OM'],  # Maç başına atılan gol
                home_stats['YG'] / home_stats['OM'],  # Maç başına yenilen gol
                home_stats['P'] / home_stats['OM'],   # Maç başına puan
                home_stats['A'],                      # Averaj
                home_last_5_goals_scored / max(len(home_last_5), 1),  # Son 5 maçta atılan gol ortalaması
                home_last_5_goals_conceded / max(len(home_last_5), 1),  # Son 5 maçta yenilen gol ortalaması
                
                away_stats['G'] / away_stats['OM'],
                away_stats['B'] / away_stats['OM'],
                away_stats['M'] / away_stats['OM'],
                away_stats['AG'] / away_stats['OM'],
                away_stats['YG'] / away_stats['OM'],
                away_stats['P'] / away_stats['OM'],
                away_stats['A'],
                away_last_5_goals_scored / max(len(away_last_5), 1),
                away_last_5_goals_conceded / max(len(away_last_5), 1),
                
                # İki takım arasındaki farklar
                (home_stats['P'] - away_stats['P']) / home_stats['OM'],  # Puan farkı
                home_stats['A'] - away_stats['A'],  # Averaj farkı
            ]
            
            features.append(match_features)
            home_goals.append(match['home_goals'])
            away_goals.append(match['away_goals'])
        
        return np.array(features), np.array(home_goals), np.array(away_goals)
    
    def train(self, matches_df, stats_df):
        X, y_home, y_away = self.create_feature_dataset(matches_df, stats_df)
        
        # Feature'ları standardize et
        X_scaled = self.scaler.fit_transform(X)
        
        # Veri setini böl
        X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
            X_scaled, y_home, y_away, test_size=0.2, random_state=42
        )
        
        # Gradient Boosting modellerini oluştur ve eğit
        self.home_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        )
        
        self.away_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        )
        
        # Modelleri eğit
        self.home_model.fit(X_train, y_home_train)
        self.away_model.fit(X_train, y_away_train)
        
        # Test seti üzerinde performansı değerlendir
        home_pred = self.home_model.predict(X_test)
        away_pred = self.away_model.predict(X_test)
        
        # Performans metriklerini hesapla
        home_mse = mean_squared_error(y_home_test, home_pred)
        away_mse = mean_squared_error(y_away_test, away_pred)
        home_mae = mean_absolute_error(y_home_test, home_pred)
        away_mae = mean_absolute_error(y_away_test, away_pred)
        
        # Cross-validation skorları
        home_cv_scores = cross_val_score(self.home_model, X_scaled, y_home, cv=5)
        away_cv_scores = cross_val_score(self.away_model, X_scaled, y_away, cv=5)
        
        return {
            'home_mse': home_mse,
            'away_mse': away_mse,
            'home_mae': home_mae,
            'away_mae': away_mae,
            'home_cv_mean': home_cv_scores.mean(),
            'away_cv_mean': away_cv_scores.mean()
        }
    
    def predict_match(self, home_team, away_team, matches_df, stats_df):
        # Son maç numarasını bul
        last_match_no = matches_df['match_no'].max()
        
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
        
        for _, last_match in home_last_5.iterrows():
            if last_match['home_team'] == home_team:
                home_last_5_goals_scored += last_match['home_goals']
                home_last_5_goals_conceded += last_match['away_goals']
            else:
                home_last_5_goals_scored += last_match['away_goals']
                home_last_5_goals_conceded += last_match['home_goals']
                
        for _, last_match in away_last_5.iterrows():
            if last_match['home_team'] == away_team:
                away_last_5_goals_scored += last_match['home_goals']
                away_last_5_goals_conceded += last_match['away_goals']
            else:
                away_last_5_goals_scored += last_match['away_goals']
                away_last_5_goals_conceded += last_match['home_goals']
        
        home_stats = stats_df[stats_df['team_name'] == home_team].iloc[0]
        away_stats = stats_df[stats_df['team_name'] == away_team].iloc[0]
        
        features = [
            home_stats['G'] / home_stats['OM'],
            home_stats['B'] / home_stats['OM'],
            home_stats['M'] / home_stats['OM'],
            home_stats['AG'] / home_stats['OM'],
            home_stats['YG'] / home_stats['OM'],
            home_stats['P'] / home_stats['OM'],
            home_stats['A'],
            home_last_5_goals_scored / len(home_last_5),
            home_last_5_goals_conceded / len(home_last_5),
            
            away_stats['G'] / away_stats['OM'],
            away_stats['B'] / away_stats['OM'],
            away_stats['M'] / away_stats['OM'],
            away_stats['AG'] / away_stats['OM'],
            away_stats['YG'] / away_stats['OM'],
            away_stats['P'] / away_stats['OM'],
            away_stats['A'],
            away_last_5_goals_scored / len(away_last_5),
            away_last_5_goals_conceded / len(away_last_5),
            
            (home_stats['P'] - away_stats['P']) / home_stats['OM'],
            home_stats['A'] - away_stats['A'],
        ]
        
        # Feature'ları standardize et
        features_scaled = self.scaler.transform([features])
        
        # Tahminleri yap
        predicted_home_goals = max(0, round(self.home_model.predict(features_scaled)[0]))
        predicted_away_goals = max(0, round(self.away_model.predict(features_scaled)[0]))
        
        return predicted_home_goals, predicted_away_goals
    
    def save_models(self, model_dir='model'):
        """Modelleri ve scaler'ı kaydet"""
        # Model klasörünü oluştur
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Modelleri kaydet
        joblib.dump(self.home_model, os.path.join(model_dir, 'home_goals_model.joblib'))
        joblib.dump(self.away_model, os.path.join(model_dir, 'away_goals_model.joblib'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))
        
        print(f"Modeller {model_dir} klasörüne kaydedildi.")
        
    def load_models(self, model_dir='model'):
        """Kaydedilmiş modelleri ve scaler'ı yükle"""
        try:
            self.home_model = joblib.load(os.path.join(model_dir, 'home_goals_model.joblib'))
            self.away_model = joblib.load(os.path.join(model_dir, 'away_goals_model.joblib'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
            print("Modeller başarıyla yüklendi.")
            return True
        except FileNotFoundError:
            print("Kaydedilmiş model dosyaları bulunamadı.")
            return False
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            return False

# Kullanım örneği
if __name__ == "__main__":
    # Model klasörünü oluştur
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Veri setlerini yükle
    matches_df = pd.read_csv('data/match_details.csv')
    stats_df = pd.read_csv('data/team_stats.csv')

    # Modeli oluştur ve eğit
    predictor = MatchPredictor()
    metrics = predictor.train(matches_df, stats_df)

    # Performans metriklerini göster
    print("\nModel Performans Metrikleri:")
    print(f"Ev Sahibi MSE: {metrics['home_mse']:.4f}")
    print(f"Deplasman MSE: {metrics['away_mse']:.4f}")
    print(f"Ev Sahibi MAE: {metrics['home_mae']:.4f}")
    print(f"Deplasman MAE: {metrics['away_mae']:.4f}")
    print(f"Ev Sahibi CV Score: {metrics['home_cv_mean']:.4f}")
    print(f"Deplasman CV Score: {metrics['away_cv_mean']:.4f}")

    # Modelleri kaydet
    predictor.save_models(model_dir)

    # Örnek tahmin
    home_goals, away_goals = predictor.predict_match('Galatasaray', 'Fenerbahçe', matches_df, stats_df)
    print(f"\nTahmin: Galatasaray {home_goals} - {away_goals} Fenerbahçe")

    # Modelleri yükleyerek test et
    new_predictor = MatchPredictor()
    if new_predictor.load_models(model_dir):
        home_goals, away_goals = new_predictor.predict_match('Galatasaray', 'Fenerbahçe', matches_df, stats_df)