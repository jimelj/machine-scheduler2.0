"""
Database schema for the Production Scheduler application.

This module defines the SQLAlchemy models for the database schema
that will support the production scheduling system.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class ZipCode(db.Model):
    """
    Model for zip codes from the Zips by Address File Group.
    Contains information about mail day and truckload assignments.
    """
    __tablename__ = 'zip_codes'
    
    id = db.Column(db.Integer, primary_key=True)
    zip = db.Column(db.String(5), unique=True, nullable=False)
    ratedesc = db.Column(db.String(10), nullable=False)
    mailday = db.Column(db.String(3), nullable=False)  # MON or TUE
    truckload = db.Column(db.String(10), nullable=False)
    
    # Relationship with orders
    orders = db.relationship('Order', backref='zip_code', lazy=True)
    
    def __repr__(self):
        return f'<ZipCode {self.zip} ({self.mailday})>'


class Advertiser(db.Model):
    """
    Model for advertisers who provide inserts.
    """
    __tablename__ = 'advertisers'
    
    id = db.Column(db.Integer, primary_key=True)
    account_name = db.Column(db.String(100), nullable=False)
    store_name = db.Column(db.String(100), nullable=True)
    
    # Relationship with orders
    orders = db.relationship('Order', backref='advertiser', lazy=True)
    
    def __repr__(self):
        return f'<Advertiser {self.account_name}>'


class Order(db.Model):
    """
    Model for insert orders filtered for CBA LONG ISLAND.
    """
    __tablename__ = 'orders'
    
    id = db.Column(db.Integer, primary_key=True)
    zip_route = db.Column(db.String(10), nullable=False)
    zip_code_id = db.Column(db.Integer, db.ForeignKey('zip_codes.id'), nullable=False)
    advertiser_id = db.Column(db.Integer, db.ForeignKey('advertisers.id'), nullable=False)
    combined_store_name = db.Column(db.String(200), nullable=False)
    pieces = db.Column(db.Integer, nullable=False)
    order_status = db.Column(db.String(20), nullable=False)
    in_home_end_date = db.Column(db.Date, nullable=False)
    zip_route_version = db.Column(db.String(10), nullable=True)
    
    # Relationship with schedule items
    schedule_items = db.relationship('ScheduleItem', backref='order', lazy=True)
    
    def __repr__(self):
        return f'<Order {self.zip_route} ({self.combined_store_name})>'


class Machine(db.Model):
    """
    Model for production machines.
    Each machine has 16 pockets that can hold inserts.
    """
    __tablename__ = 'machines'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    pocket_count = db.Column(db.Integer, default=16, nullable=False)
    
    # Relationship with schedule items
    schedule_items = db.relationship('ScheduleItem', backref='machine', lazy=True)
    
    # Relationship with machine pockets
    pockets = db.relationship('MachinePocket', backref='machine', lazy=True)
    
    def __repr__(self):
        return f'<Machine {self.name} ({self.pocket_count} pockets)>'


class MachinePocket(db.Model):
    """
    Model for machine pockets.
    Each pocket can hold one insert type.
    """
    __tablename__ = 'machine_pockets'
    
    id = db.Column(db.Integer, primary_key=True)
    machine_id = db.Column(db.Integer, db.ForeignKey('machines.id'), nullable=False)
    pocket_number = db.Column(db.Integer, nullable=False)
    current_insert = db.Column(db.String(100), nullable=True)
    
    # Composite unique constraint to ensure each pocket number is unique per machine
    __table_args__ = (
        db.UniqueConstraint('machine_id', 'pocket_number', name='unique_pocket_per_machine'),
    )
    
    def __repr__(self):
        return f'<Pocket {self.pocket_number} on Machine {self.machine_id}>'


class ScheduleItem(db.Model):
    """
    Model for production schedule items.
    Represents the assignment of an order to a specific machine.
    """
    __tablename__ = 'schedule_items'
    
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False)
    machine_id = db.Column(db.Integer, db.ForeignKey('machines.id'), nullable=False)
    sequence_number = db.Column(db.Integer, nullable=False)
    scheduled_date = db.Column(db.Date, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Composite unique constraint to ensure each order is scheduled only once
    __table_args__ = (
        db.UniqueConstraint('order_id', name='unique_order_schedule'),
    )
    
    # Relationship with pocket assignments
    pocket_assignments = db.relationship('PocketAssignment', backref='schedule_item', lazy=True)
    
    def __repr__(self):
        return f'<ScheduleItem {self.id} (Machine {self.machine_id}, Sequence {self.sequence_number})>'


class PocketAssignment(db.Model):
    """
    Model for pocket assignments.
    Represents which insert is assigned to which pocket for a specific schedule item.
    """
    __tablename__ = 'pocket_assignments'
    
    id = db.Column(db.Integer, primary_key=True)
    schedule_item_id = db.Column(db.Integer, db.ForeignKey('schedule_items.id'), nullable=False)
    pocket_id = db.Column(db.Integer, db.ForeignKey('machine_pockets.id'), nullable=False)
    insert_name = db.Column(db.String(100), nullable=False)
    
    # Composite unique constraint to ensure each pocket is assigned only once per schedule item
    __table_args__ = (
        db.UniqueConstraint('schedule_item_id', 'pocket_id', name='unique_pocket_assignment'),
    )
    
    def __repr__(self):
        return f'<PocketAssignment {self.id} (Insert: {self.insert_name})>'


class ScheduleRun(db.Model):
    """
    Model for schedule runs.
    Represents a complete scheduling run with metadata.
    """
    __tablename__ = 'schedule_runs'
    
    id = db.Column(db.Integer, primary_key=True)
    run_date = db.Column(db.DateTime, default=datetime.utcnow)
    description = db.Column(db.String(200), nullable=True)
    total_orders = db.Column(db.Integer, default=0)
    total_inserts_reused = db.Column(db.Integer, default=0)
    
    def __repr__(self):
        return f'<ScheduleRun {self.id} ({self.run_date})>'
