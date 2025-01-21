from app import create_app, db
from app.models import Team, Match
import pandas as pd

app = create_app()

def init_db():
    with app.app_context():
        db.create_all()
        
        # Eğer veritabanı boşsa, CSV'den verileri yükle
        if Team.query.count() == 0:
            # Takım istatistiklerini yükle
            teams_df = pd.read_csv('data/team_stats.csv')
            
            for _, row in teams_df.iterrows():
                team = Team(
                    team_name=row['team_name'],
                    OM=row['OM'],
                    G=row['G'],
                    B=row['B'],
                    M=row['M'],
                    P=row['P'],
                    AG=row['AG'],
                    YG=row['YG'],
                    A=row['A']
                )
                db.session.add(team)
            
            # Maç verilerini yükle
            matches_df = pd.read_csv('data/match_details.csv')
            team_dict = {team.team_name: team.id for team in Team.query.all()}
            
            for _, row in matches_df.iterrows():
                match = Match(
                    home_team_id=team_dict[row['home_team']],
                    away_team_id=team_dict[row['away_team']],
                    home_goals=row['home_goals'],
                    away_goals=row['away_goals'],
                    match_no=row['match_no']
                )
                db.session.add(match)
            
            db.session.commit()
            print("Veriler başarıyla yüklendi!")

if __name__ == '__main__':
    init_db()
    app.run(debug=True)