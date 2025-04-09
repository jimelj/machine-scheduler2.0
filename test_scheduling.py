"""
Test script for the Production Scheduler application.

This script tests the scheduling algorithm and application functionality.
"""

import os
import sys
import unittest
import json
import pandas as pd
from datetime import datetime

# Add the parent directory to the path so we can import the application modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduling_algorithm import SchedulingAlgorithm, load_and_prepare_data, run_scheduling

class TestSchedulingAlgorithm(unittest.TestCase):
    """Test cases for the scheduling algorithm."""
    
    def setUp(self):
        """Set up test data."""
        self.zip_file_path = "/home/ubuntu/upload/Zips by Address File Group.xlsx"
        self.orders_file_path = "/home/ubuntu/upload/Insert Orders IHD 4.12.25.csv"
        
        # Load test data
        self.zip_codes_df, self.orders_df = load_and_prepare_data(
            self.zip_file_path, 
            self.orders_file_path
        )
        
        # Initialize the scheduler
        self.scheduler = SchedulingAlgorithm()
        self.scheduler.load_data(self.zip_codes_df, self.orders_df)
    
    def test_data_loading(self):
        """Test that data is loaded correctly."""
        # Check that zip codes are loaded
        self.assertGreater(len(self.zip_codes_df), 0)
        self.assertIn('zip', self.zip_codes_df.columns)
        self.assertIn('mailday', self.zip_codes_df.columns)
        
        # Check that orders are loaded and filtered
        self.assertGreater(len(self.orders_df), 0)
        self.assertIn('ZipRoute', self.orders_df.columns)
        self.assertIn('CombinedStoreName', self.orders_df.columns)
        self.assertIn('ZipCode', self.orders_df.columns)
        
        # Check that all orders are for CBA LONG ISLAND
        self.assertTrue(all(self.orders_df['DistributorName'] == 'CBA LONG ISLAND'))
    
    def test_overlap_calculation(self):
        """Test the overlap calculation between zip codes."""
        # Get some sample zip codes
        zip_codes = list(self.scheduler.inserts_by_zip.keys())
        if len(zip_codes) >= 2:
            zip1, zip2 = zip_codes[0], zip_codes[1]
            
            # Calculate overlap
            overlap = self.scheduler.calculate_overlap(zip1, zip2)
            
            # Check that the overlap is a non-negative integer
            self.assertIsInstance(overlap, int)
            self.assertGreaterEqual(overlap, 0)
    
    def test_schedule_production(self):
        """Test the schedule production function."""
        # Generate a schedule for Monday
        schedule_mon = self.scheduler.schedule_production('MON')
        
        # Check that the schedule has the expected structure
        self.assertIn('machine_assignments', schedule_mon)
        self.assertIn('total_inserts_reused', schedule_mon)
        self.assertIn('total_zip_codes', schedule_mon)
        
        # Check that machines are assigned
        self.assertGreater(len(schedule_mon['machine_assignments']), 0)
        
        # Generate a schedule for Tuesday
        schedule_tue = self.scheduler.schedule_production('TUE')
        
        # Check that the schedule has the expected structure
        self.assertIn('machine_assignments', schedule_tue)
        self.assertIn('total_inserts_reused', schedule_tue)
        self.assertIn('total_zip_codes', schedule_tue)
        
        # Check that machines are assigned
        self.assertGreater(len(schedule_tue['machine_assignments']), 0)
    
    def test_pocket_changes(self):
        """Test the pocket changes generation."""
        # Generate a schedule
        schedule = self.scheduler.schedule_production()
        
        # Generate pocket changes
        pocket_changes = self.scheduler.generate_pocket_changes(schedule)
        
        # Check that pocket changes are generated for each machine
        for machine_id in schedule['machine_assignments']:
            self.assertIn(machine_id, pocket_changes)
            
            # Check that changes have the expected structure
            for change in pocket_changes[machine_id]:
                self.assertIn('zip_code', change)
                self.assertIn('sequence', change)
                self.assertIn('action', change)
                self.assertIn('pocket', change)
                self.assertIn('insert', change)
                
                # Check that action is either 'add' or 'remove'
                self.assertIn(change['action'], ['add', 'remove'])
                
                # Check that pocket is between 1 and 16
                self.assertGreaterEqual(change['pocket'], 1)
                self.assertLessEqual(change['pocket'], 16)
    
    def test_run_scheduling(self):
        """Test the run_scheduling function."""
        # Run scheduling for all days
        schedule = run_scheduling(self.zip_file_path, self.orders_file_path)
        
        # Check that the schedule has the expected structure
        self.assertIn('machine_assignments', schedule)
        self.assertIn('total_inserts_reused', schedule)
        self.assertIn('total_zip_codes', schedule)
        self.assertIn('pocket_changes', schedule)
        
        # Check that machines are assigned
        self.assertGreater(len(schedule['machine_assignments']), 0)
        
        # Check that pocket changes are generated
        self.assertGreater(len(schedule['pocket_changes']), 0)
        
        # Run scheduling for Monday only
        schedule_mon = run_scheduling(self.zip_file_path, self.orders_file_path, 'MON')
        
        # Check that the schedule has the expected structure
        self.assertIn('machine_assignments', schedule_mon)
        self.assertIn('total_inserts_reused', schedule_mon)
        self.assertIn('total_zip_codes', schedule_mon)
        self.assertIn('pocket_changes', schedule_mon)
        
        # Check that machines are assigned
        self.assertGreater(len(schedule_mon['machine_assignments']), 0)
        
        # Run scheduling for Tuesday only
        schedule_tue = run_scheduling(self.zip_file_path, self.orders_file_path, 'TUE')
        
        # Check that the schedule has the expected structure
        self.assertIn('machine_assignments', schedule_tue)
        self.assertIn('total_inserts_reused', schedule_tue)
        self.assertIn('total_zip_codes', schedule_tue)
        self.assertIn('pocket_changes', schedule_tue)
        
        # Check that machines are assigned
        self.assertGreater(len(schedule_tue['machine_assignments']), 0)
    
    def test_continuity_optimization(self):
        """Test that the algorithm optimizes for insert continuity."""
        # Generate a schedule
        schedule = self.scheduler.schedule_production()
        
        # Check that inserts are reused
        self.assertGreater(schedule['total_inserts_reused'], 0)
        
        # Check that each machine has some inserts reused
        for machine_id, assignments in schedule['machine_assignments'].items():
            if len(assignments) > 1:  # Only check if there are at least 2 assignments
                total_reused = sum(assignment['inserts_reused'] for assignment in assignments[1:])
                self.assertGreater(total_reused, 0, f"Machine {machine_id} has no inserts reused")

if __name__ == '__main__':
    unittest.main()
