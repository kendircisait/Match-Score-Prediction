from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
import random
import pandas as pd
import os
from datetime import datetime, timedelta

Base = declarative_base()

class TeamStats(Base):
    __tablename__ = 'team_stats'

    id = Column(Integer, primary_key=True)
    team_name = Column(String(255), nullable=False)
    OM = Column(Integer, nullable=False)
    G = Column(Integer, nullable=False)
    B = Column(Integer, nullable=False)
    M = Column(Integer, nullable=False)
    P = Column(Integer, nullable=False)
    AG = Column(Integer, nullable=False)
    YG = Column(Integer, nullable=False)
    A = Column(Integer, nullable=False)

class Match(Base):
    __tablename__ = 'matches'

    id = Column(Integer, primary_key=True)
    home_team = Column(String(255), nullable=False)
    away_team = Column(String(255), nullable=False)
    home_goals = Column(Integer, nullable=False)
    away_goals = Column(Integer, nullable=False)
    match_no = Column(Integer, nullable=False)
    match_date = Column(DateTime, default=datetime.utcnow)

# Veritabanı bağlantısı
engine = create_engine('sqlite:///team_stats.db')
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

teams = [
    "Adana Demirspor", "Alanyaspor", "Antalyaspor", "Beşiktaş", "Bodrum",
    "Çaykur Rizespor", "Eyüpspor", "Fenerbahçe", "Galatasaray", "Gaziantep",
    "Göztepe", "Hatayspor", "İstanbul Başakşehir", "Kasımpaşa", "Kayserispor",
    "Konyaspor", "Samsunspor", "Sivasspor", "Trabzonspor"
]

# Takım güç seviyeleri
team_strengths = {
    "Galatasaray": 9.85,
    "Fenerbahçe": 9.45,
    "Beşiktaş": 8.85,
    "Trabzonspor": 8.5
}

default_strength = 6.0

def weighted_random_goal(home_team, away_team, is_home=True):
    home_strength = team_strengths.get(home_team, default_strength)
    away_strength = team_strengths.get(away_team, default_strength)
    
    # Ev sahibi avantajı
    home_advantage = 1.3 if is_home else 1.0
    
    base_strength = home_strength * home_advantage if is_home else away_strength
    opponent_strength = away_strength if is_home else home_strength
    
    # Güç farkına göre gol atma olasılıkları
    strength_diff = base_strength - opponent_strength
    
    if home_team in ["Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor"]:
        # Bu takımlar için galibiyet olasılığı daha yüksek
        if strength_diff > 2:
            weights = [10, 25, 40, 20, 5]  # 2-3 goller artırıldı, 4 gol düşürüldü
        elif strength_diff > 0:
            weights = [15, 30, 35, 15, 5]
        else:
            weights = [25, 35, 30, 8, 2]
    else:
        # Diğer takımlar için standart ağırlıklar
        if strength_diff > 2:
            weights = [15, 30, 35, 15, 5]
        elif strength_diff > 0:
            weights = [20, 35, 30, 10, 5]
        elif strength_diff > -2:
            weights = [30, 35, 25, 8, 2]
        else:
            weights = [40, 35, 20, 4, 1]
    
    return random.choices([0, 1, 2, 3, 4], weights=weights)[0]


def adjust_wins(results):
    # Belirli takımların galibiyetlerini artır
    target_win_percentages = {
        "Galatasaray": 90,
        "Fenerbahçe": 88,
        "Beşiktaş": 85,
        "Trabzonspor": 80
    }

    for team, target_percentage in target_win_percentages.items():
        total_matches = results[team]["OM"]
        target_wins = int(total_matches * (target_percentage / 100))
        current_wins = results[team]["G"]
        additional_wins = target_wins - current_wins

        if additional_wins > 0:
            # Puan ve galibiyet sayısını artır
            results[team]["G"] += additional_wins
            results[team]["P"] += additional_wins * 3
            results[team]["B"] = max(0, results[team]["B"] - additional_wins)
            results[team]["M"] = max(0, results[team]["M"] - additional_wins)
    
    return results

def generate_fake_match_data():
    results = {team: {"OM": 0, "G": 0, "B": 0, "M": 0, "P": 0, "AG": 0, "YG": 0} for team in teams}
    matches = []
    
    start_date = datetime(2024, 1, 1)
    
    for team in teams:
        for opponent in teams:
            if team != opponent:
                for match_no in range(10):
                    # Gol tahminleri
                    home_goals = weighted_random_goal(team, opponent, True)
                    away_goals = weighted_random_goal(team, opponent, False)
                    
                    # Maç tarihi
                    match_date = start_date + timedelta(days=match_no*7)
                    
                    # Maç detaylarını kaydet
                    match = Match(
                        home_team=team,
                        away_team=opponent,
                        home_goals=home_goals,
                        away_goals=away_goals,
                        match_no=match_no + 1,
                        match_date=match_date
                    )
                    session.add(match)
                    
                    matches.append({
                        'home_team': team,
                        'away_team': opponent,
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'match_no': match_no + 1
                    })
                    
                    # İstatistikleri güncelle
                    results[team]["OM"] += 1
                    results[team]["AG"] += home_goals
                    results[team]["YG"] += away_goals
                    
                    if home_goals > away_goals:
                        results[team]["G"] += 1
                        results[team]["P"] += 3
                    elif home_goals < away_goals:
                        results[team]["M"] += 1
                    else:
                        results[team]["B"] += 1
                        results[team]["P"] += 1

    # Galibiyet oranlarını ayarla
    results = adjust_wins(results)

    # Takım istatistiklerini kaydet
    for team, stats in results.items():
        stats["A"] = stats["AG"] - stats["YG"]
        team_stat = TeamStats(
            team_name=team,
            OM=stats["OM"],
            G=stats["G"],
            B=stats["B"],
            M=stats["M"],
            P=stats["P"],
            AG=stats["AG"],
            YG=stats["YG"],
            A=stats["A"]
        )
        session.add(team_stat)
    
    session.commit()
    return matches

def save_to_csv():
    # Data klasörünü oluştur
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Takım istatistiklerini CSV'ye kaydet
    team_stats_df = pd.read_sql('SELECT * FROM team_stats', engine)
    team_stats_df = team_stats_df.drop('id', axis=1)
    team_stats_df.to_csv(os.path.join(data_dir, 'team_stats.csv'), index=False)

    # Maç detaylarını CSV'ye kaydet
    matches_df = pd.read_sql('SELECT * FROM matches', engine)
    matches_df = matches_df.drop('id', axis=1)
    matches_df.to_csv(os.path.join(data_dir, 'match_details.csv'), index=False)

    return team_stats_df, matches_df

if __name__ == "__main__":
    print("Fake maç verileri oluşturuluyor...")
    matches = generate_fake_match_data()
    
    print("Veriler CSV dosyalarına kaydediliyor...")
    team_stats_df, matches_df = save_to_csv()
    
    print("\nTakım İstatistikleri Önizleme:")
    print(team_stats_df.sort_values('P', ascending=False)[['team_name', 'P', 'G', 'B', 'M', 'A']].head())
    
    print("\nMaç Detayları Önizleme:")
    print(matches_df.head())
    
    print("\nİşlem tamamlandı!")
    print(f"Toplam {len(matches)} maç kaydedildi.")
