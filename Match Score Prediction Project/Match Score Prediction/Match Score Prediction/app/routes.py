from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.models import Team, Match, db
from app.predictor import MatchPredictor
import joblib
import os
import pandas as pd

main = Blueprint('main', __name__)

@main.route('/')
def home():
    teams = Team.query.order_by(Team.P.desc(), Team.A.desc()).all()
    return render_template('home.html', teams=teams)
    
@main.route('/predict', methods=['GET', 'POST'])
def predict():
    teams = Team.query.all()
    if request.method == 'POST':
        home_team_id = request.form.get('home_team')
        away_team_id = request.form.get('away_team')
        
        if home_team_id == away_team_id:
            flash('Aynı takımı seçemezsiniz!', 'danger')
            return redirect(url_for('main.predict'))
        
        home_team = Team.query.get(home_team_id)
        away_team = Team.query.get(away_team_id)
        
        if home_team and away_team:
            try:
                # Debug için print ekleyelim
                print(f"Ev sahibi: {home_team.team_name}")
                print(f"Deplasman: {away_team.team_name}")
                
                predictor = MatchPredictor()
                model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
                if not predictor.load_models(model_dir):
                    flash('Model dosyaları yüklenemedi!', 'danger')
                    return redirect(url_for('main.predict'))
                
                # Maç verilerini çekelim
                matches = Match.query.order_by(Match.match_date.desc()).all()
                if not matches:
                    # Eğer hiç maç yoksa, CSV'den okuyalım
                    matches_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'match_details.csv'))
                else:
                    matches_data = []
                    for match in matches:
                        matches_data.append({
                            'home_team': match.home_team.team_name,
                            'away_team': match.away_team.team_name,
                            'home_goals': match.home_goals,
                            'away_goals': match.away_goals,
                            'match_no': match.match_no
                        })
                    matches_df = pd.DataFrame(matches_data)
                
                # Debug için print ekleyelim
                print("Maç verileri yüklendi:")
                print(matches_df.head())
                
                # Takım istatistiklerini DataFrame'e çevir
                stats_data = [{
                    'team_name': team.team_name,
                    'OM': team.OM,
                    'G': team.G,
                    'B': team.B,
                    'M': team.M,
                    'P': team.P,
                    'AG': team.AG,
                    'YG': team.YG,
                    'A': team.A
                } for team in teams]
                stats_df = pd.DataFrame(stats_data)
                
                # Debug için print ekleyelim
                print("Takım istatistikleri yüklendi:")
                print(stats_df.head())
                
                # Tahmin yap
                predicted_home_goals, predicted_away_goals = predictor.predict_match(
                    home_team.team_name,
                    away_team.team_name,
                    matches_df,
                    stats_df
                )
                
                return render_template('predict.html', 
                                    teams=teams,
                                    prediction_made=True,
                                    home_team=home_team,
                                    away_team=away_team,
                                    home_goals=predicted_home_goals,
                                    away_goals=predicted_away_goals,
                                    raw_home_goals=predicted_home_goals,
                                    raw_away_goals=predicted_away_goals)
                
            except Exception as e:
                import traceback
                print(f"Hata detayı:")
                print(traceback.format_exc())
                flash(f'Tahmin yapılırken bir hata oluştu: {str(e)}', 'danger')
                return redirect(url_for('main.predict'))
            
    return render_template('predict.html', teams=teams, prediction_made=False)

@main.route('/match', methods=['GET', 'POST'])
def match():
    teams = Team.query.all()
    if request.method == 'POST':
        home_team_id = request.form.get('home_team')
        away_team_id = request.form.get('away_team')
        home_goals = int(request.form.get('home_goals'))
        away_goals = int(request.form.get('away_goals'))
        
        if home_team_id == away_team_id:
            flash('Aynı takımı seçemezsiniz!', 'danger')
            return redirect(url_for('main.match'))
        
        home_team = Team.query.get(home_team_id)
        away_team = Team.query.get(away_team_id)
        
        if home_team and away_team:
            # İstatistikleri güncelle
            home_team.update_after_match(home_goals, away_goals)
            away_team.update_after_match(away_goals, home_goals)
            
            # Yeni maçı kaydet
            last_match = Match.query.order_by(Match.match_no.desc()).first()
            new_match_no = (last_match.match_no + 1) if last_match else 1
            
            new_match = Match(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_goals=home_goals,
                away_goals=away_goals,
                match_no=new_match_no
            )
            db.session.add(new_match)
            db.session.commit()
            
            flash('Maç sonucu başarıyla kaydedildi!', 'success')
            return redirect(url_for('main.home'))
        
    return render_template('match.html', teams=teams)

@main.route('/history')
def history():
    # Geçmiş verileri CSV'den oku
    df = pd.read_csv('data/team_stats.csv')
    historical_teams = df.to_dict('records')
    return render_template('history.html', teams=historical_teams)