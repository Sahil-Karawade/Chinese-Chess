# models.py
from .sql_db import db
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime

class Game(db.Model):
    __tablename__ = 'games'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, nullable=False)
    result = db.Column(db.String(10))  # e.g., 'win', 'loss', 'draw'
    source_file = db.Column(db.String())
    samples = db.relationship('Sample', backref='game', lazy=True)
    z_col = db.Column(db.Integer)

class Sample(db.Model):
    __tablename__ = 'samples'
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.Integer, db.ForeignKey('games.id'), nullable=False)

    state = db.Column(db.LargeBinary, nullable=False)    # store game state as JSON
    pi = db.Column(db.LargeBinary, nullable=False)       # policy vector as JSON
    z = db.Column(db.Float)     # scalar value
    entropy = db.Column(db.Float)        #  Confidence of the policy
    best_move_prob = db.Column(db.Float) #  Top move probability
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class MultiGame(db.Model):
    __tablename__ = 'MultiPlayerGame'
    id = db.Column(db.Integer, primary_key=True, index=True, autoincrement=True)
    player_red = db.Column(db.String(), nullable=True)
    player_black = db.Column(db.String(), nullable=True)
    current_turn = db.Column(db.String(), nullable=False, default="red")
    winner = db.Column(db.String(), nullable=True)
    game_over = db.Column(db.Boolean, nullable=False, default=False)
    board = db.Column(JSONB, nullable=False)
    timed = db.Column(db.Boolean, default=False)
    time_limit = db.Column(db.Integer, nullable=True)      # seconds per player
    red_time_left = db.Column(db.Integer, nullable=True, default=600)
    black_time_left = db.Column(db.Integer, nullable=True, default=600)
    last_move_time = db.Column(db.DateTime, default=datetime.utcnow)
