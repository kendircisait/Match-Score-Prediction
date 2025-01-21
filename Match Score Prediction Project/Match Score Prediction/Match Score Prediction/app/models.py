from app import db
from datetime import datetime  # Bu satırı ekleyelim

class Team(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    team_name = db.Column(db.String(100), nullable=False)
    OM = db.Column(db.Integer, default=0)
    G = db.Column(db.Integer, default=0)
    B = db.Column(db.Integer, default=0)
    M = db.Column(db.Integer, default=0)
    P = db.Column(db.Integer, default=0)
    AG = db.Column(db.Integer, default=0)
    YG = db.Column(db.Integer, default=0)
    A = db.Column(db.Integer, default=0)

    def update_after_match(self, goals_scored, goals_conceded):
        self.OM += 1
        self.AG += goals_scored
        self.YG += goals_conceded
        self.A = self.AG - self.YG
        
        if goals_scored > goals_conceded:
            self.G += 1
            self.P += 3
        elif goals_scored < goals_conceded:
            self.M += 1
        else:
            self.B += 1
            self.P += 1

class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    home_team_id = db.Column(db.Integer, db.ForeignKey('team.id'), nullable=False)
    away_team_id = db.Column(db.Integer, db.ForeignKey('team.id'), nullable=False)
    home_goals = db.Column(db.Integer, nullable=False)
    away_goals = db.Column(db.Integer, nullable=False)
    match_no = db.Column(db.Integer, nullable=False)
    match_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    home_team = db.relationship('Team', foreign_keys=[home_team_id])
    away_team = db.relationship('Team', foreign_keys=[away_team_id])