"""
Additional debugging script to identify issues with the scheduling algorithm.
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime

# Add the parent directory to the path so we can import the application modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduling_algorithm import SchedulingAlgorithm, load_and_prepare_data, run_scheduling

def debug_scheduling_algorithm_detailed():
    """Debug the scheduling algorithm with more detailed output."""
    print("Starting detailed debugging of scheduling algorithm...")
    
    # Load test data
    zip_file_path = "upload/Zips by Address File Group.xlsx"
    orders_file_path = "upload/Insert Orders IHD 4.12.25.csv"
    
    print("Loading data...")
    zip_codes_df, orders_df = load_and_prepare_data(zip_file_path, orders_file_path)
    
    print(f"Loaded {len(zip_codes_df)} zip codes and {len(orders_df)} orders")
    
    # Initialize the scheduler
    print("\nInitializing scheduler...")
    scheduler = SchedulingAlgorithm()
    
    # Load data into the scheduler
    print("Loading data into scheduler...")
    scheduler.load_data(zip_codes_df, orders_df)
    
    # Check if inserts_by_zip is populated in the scheduler
    print(f"Number of zip codes with inserts in scheduler: {len(scheduler.inserts_by_zip)}")
    
    # Print the first few zip codes in the zip_codes_df
    print("\nFirst 5 zip codes from zip_codes_df:")
    for i, row in zip_codes_df.head().iterrows():
        print(f"Zip: {row['zip']}, Mailday: {row['mailday']}")
    
    # Check if these zip codes are in inserts_by_zip
    print("\nChecking if these zip codes have inserts:")
    for i, row in zip_codes_df.head().iterrows():
        zip_code = row['zip']
        if zip_code in scheduler.inserts_by_zip:
            print(f"Zip {zip_code}: {len(scheduler.inserts_by_zip[zip_code])} inserts")
        else:
            print(f"Zip {zip_code}: No inserts found")
    
    # Check data types
    print("\nChecking data types:")
    print(f"Type of zip in zip_codes_df: {zip_codes_df['zip'].dtype}")
    print(f"Type of ZipCode in orders_df: {orders_df['ZipCode'].dtype}")
    
    # Convert zip codes to strings for comparison
    print("\nConverting zip codes to strings and checking again:")
    zip_codes_str = [str(zip_code) for zip_code in zip_codes_df['zip']]
    order_zips_str = [str(zip_code) for zip_code in orders_df['ZipCode'].unique()]
    
    print(f"Sample zip codes from zip_codes_df: {zip_codes_str[:5]}")
    print(f"Sample zip codes from orders_df: {order_zips_str[:5]}")
    
    # Check for matching zip codes
    matching_zips = set(zip_codes_str).intersection(set(order_zips_str))
    print(f"Number of matching zip codes: {len(matching_zips)}")
    print(f"Sample matching zip codes: {list(matching_zips)[:5]}")
    
    # Try to generate a schedule with explicit type conversion
    print("\nGenerating schedule with explicit type conversion...")
    
    # Create a modified version of the schedule_production method for debugging
    def debug_schedule_production(day=None):
        # Reset machine states
        scheduler.initialize_machines()
        
        # Filter zip codes by day if specified
        if day:
            zip_codes_to_schedule = zip_codes_df[zip_codes_df['mailday'] == day]['zip'].astype(str).tolist()
        else:
            zip_codes_to_schedule = zip_codes_df['zip'].astype(str).tolist()
        
        print(f"Number of zip codes to schedule: {len(zip_codes_to_schedule)}")
        print(f"Sample zip codes to schedule: {zip_codes_to_schedule[:5]}")
        
        # Create a schedule
        schedule = {
            'machine_assignments': {},
            'total_inserts_reused': 0,
            'total_zip_codes': len(zip_codes_to_schedule),
            'schedule_date': datetime.now().strftime('%Y-%m-%d'),
            'day': day
        }
        
        # Track inserts reused
        total_inserts_reused = 0
        
        # Initialize machine assignments in the schedule
        for machine_id in range(1, scheduler.machines_count + 1):
            schedule['machine_assignments'][machine_id] = []
        
        # Sort zip codes by number of inserts (descending) to distribute complex zips first
        zip_codes_with_inserts = []
        for zip_code in zip_codes_to_schedule:
            if zip_code in scheduler.inserts_by_zip:
                insert_count = len(scheduler.inserts_by_zip[zip_code])
                zip_codes_with_inserts.append((zip_code, insert_count))
        
        # Sort by insert count descending
        zip_codes_with_inserts.sort(key=lambda x: x[1], reverse=True)
        
        # Start with an even distribution of the first few zips to all machines
        # This ensures all machines get used from the beginning
        for i, (zip_code, _) in enumerate(zip_codes_with_inserts[:scheduler.machines_count]):
            machine_id = i + 1  # Assign to machines 1, 2, 3
            state = scheduler.assign_zip_to_machine(zip_code, machine_id)
            
            schedule['machine_assignments'][machine_id].append({
                'zip_code': zip_code,
                'inserts': list(scheduler.inserts_by_zip[zip_code]),
                'inserts_reused': 0,
                'sequence': len(state['scheduled_zip_codes'])
            })
        
        # Process remaining zip codes using compatibility scoring
        for zip_code, _ in zip_codes_with_inserts[scheduler.machines_count:]:
            # Calculate compatibility with each machine
            compatibility = scheduler.calculate_machine_compatibility(zip_code)
            
            # Find the machine with the highest compatibility
            best_machine_id = max(compatibility, key=compatibility.get)
            overlap_count = compatibility[best_machine_id]
            
            # Assign the zip code to the best machine
            state = scheduler.assign_zip_to_machine(zip_code, best_machine_id)
            
            schedule['machine_assignments'][best_machine_id].append({
                'zip_code': zip_code,
                'inserts': list(scheduler.inserts_by_zip[zip_code]),
                'inserts_reused': overlap_count,
                'sequence': len(state['scheduled_zip_codes'])
            })
            
            # Update total inserts reused
            total_inserts_reused += overlap_count
        
        # Balance machine loads if needed
        # This is a simplified version of _balance_machine_loads
        total_assignments = sum(len(assignments) for assignments in schedule['machine_assignments'].values())
        avg_assignments = total_assignments / scheduler.machines_count
        print(f"Average assignments per machine: {avg_assignments}")
        
        schedule['total_inserts_reused'] = total_inserts_reused
        
        print(f"Zip codes with inserts: {len(zip_codes_with_inserts)}")
        print(f"Machine assignments: {len(schedule['machine_assignments'])}")
        print(f"Total inserts reused: {schedule['total_inserts_reused']}")
        
        return schedule
    
    # Run the debug version of schedule_production
    schedule = debug_schedule_production()
    
    # Check machine assignments
    for machine_id, assignments in schedule['machine_assignments'].items():
        print(f"Machine {machine_id}: {len(assignments)} assignments")
        for assignment in assignments:
            print(f"  Zip code: {assignment['zip_code']}, Inserts: {len(assignment['inserts'])}, Reused: {assignment['inserts_reused']}")
    
    print("\nDetailed debugging complete.")

if __name__ == "__main__":
    debug_scheduling_algorithm_detailed()
