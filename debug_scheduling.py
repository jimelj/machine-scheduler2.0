"""
Debug script to identify issues with the scheduling algorithm.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add the parent directory to the path so we can import the application modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduling_algorithm import SchedulingAlgorithm, load_and_prepare_data, schedule_machines

def debug_scheduling_algorithm():
    """Debug the scheduling algorithm to identify issues."""
    print("Starting debugging of scheduling algorithm...")
    
    # Load test data
    zip_file_path = "upload/Zips by Address File Group.xlsx"
    orders_file_path = "upload/Insert Orders IHD 4.12.25.csv"
    
    print("Loading data...")
    zip_codes_df, orders_df = load_and_prepare_data(zip_file_path, orders_file_path)
    
    print(f"Loaded {len(zip_codes_df)} zip codes and {len(orders_df)} orders")
    
    # Check if there are any NaN values in CombinedStoreName
    nan_count = orders_df['CombinedStoreName'].isna().sum()
    print(f"Number of NaN values in CombinedStoreName: {nan_count}")
    
    # Check if there are any zip codes with inserts
    inserts_by_zip = {}
    for zip_code, group in orders_df.groupby('ZipCode'):
        inserts = set(group['CombinedStoreName'].dropna().unique())
        inserts_by_zip[zip_code] = inserts
        
    print(f"Number of zip codes with inserts: {len(inserts_by_zip)}")
    
    # Print some sample zip codes and their inserts
    print("\nSample zip codes and their inserts:")
    count = 0
    for zip_code, inserts in inserts_by_zip.items():
        if count < 5 and len(inserts) > 0:
            print(f"Zip code {zip_code}: {len(inserts)} inserts")
            print(f"Sample inserts: {list(inserts)[:3]}")
            count += 1
    
    # Initialize the scheduler
    print("\nInitializing scheduler...")
    scheduler = SchedulingAlgorithm()
    
    # Load data into the scheduler
    print("Loading data into scheduler...")
    scheduler.load_data(zip_codes_df, orders_df)
    
    # Check if inserts_by_zip is populated in the scheduler
    print(f"Number of zip codes with inserts in scheduler: {len(scheduler.inserts_by_zip)}")
    
    # Check if there are any zip codes with inserts
    has_inserts = False
    for zip_code, inserts in scheduler.inserts_by_zip.items():
        if len(inserts) > 0:
            has_inserts = True
            print(f"Zip code {zip_code} has {len(inserts)} inserts")
            break
    
    if not has_inserts:
        print("ERROR: No zip codes have inserts!")
    
    # Try to generate a schedule
    print("\nGenerating schedule...")
    schedule = scheduler.schedule_production()
    
    print(f"Schedule generated with {len(schedule['machine_assignments'])} machine assignments")
    print(f"Total inserts reused: {schedule['total_inserts_reused']}")
    
    # Check machine assignments
    for machine_id, assignments in schedule['machine_assignments'].items():
        print(f"Machine {machine_id}: {len(assignments)} assignments")
        for assignment in assignments:
            print(f"  Zip code: {assignment['zip_code']}, Inserts: {len(assignment['inserts'])}, Reused: {assignment['inserts_reused']}")
    
    print("\nDebugging complete.")

if __name__ == "__main__":
    debug_scheduling_algorithm()
