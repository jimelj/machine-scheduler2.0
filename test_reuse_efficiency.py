"""
Test script to evaluate insert reuse efficiency across machines.
"""

import os
import json
from scheduling_algorithm import load_and_prepare_data, run_scheduling

def calculate_reuse_metrics():
    """Calculate and display detailed reuse efficiency metrics."""
    # Define file paths
    zip_file_path = "upload/Zips by Address File Group.xlsx"
    orders_file_path = "upload/Insert Orders IHD 4.12.25.csv"
    
    print("Running scheduling algorithm...")
    schedule = run_scheduling(zip_file_path, orders_file_path)
    
    print("\nReuse Efficiency Metrics:")
    print("-" * 50)
    
    # Calculate total metrics
    total_zip_codes = sum(len(assignments) for machine_id, assignments in schedule['machine_assignments'].items())
    total_inserts = 0
    total_reused = schedule['total_inserts_reused']
    potential_total = 0  # Maximum theoretical inserts that could be reused
    
    # Machine-specific metrics
    for machine_id, assignments in schedule['machine_assignments'].items():
        machine_zip_count = len(assignments)
        if machine_zip_count <= 1:
            continue  # Skip machines with only one zip code
            
        print(f"\nMachine {machine_id}:")
        print(f"  ZIP codes assigned: {machine_zip_count}")
        
        # Count inserts and reuse
        machine_inserts = 0
        machine_reused = 0
        transitions = 0
        
        for i, assignment in enumerate(assignments):
            inserts_count = len(assignment['inserts'])
            machine_inserts += inserts_count
            
            # Skip the first assignment as it has no reuse
            if i > 0:
                reused = assignment['inserts_reused'] 
                machine_reused += reused
                transitions += 1
                
                # Calculate efficiency for this transition
                previous_inserts = len(assignments[i-1]['inserts'])
                max_possible = min(previous_inserts, inserts_count)
                efficiency = (reused / max_possible) * 100 if max_possible > 0 else 0
                
                print(f"  Transition {i}: {assignments[i-1]['zip_code']} → {assignment['zip_code']}")
                print(f"    Previous inserts: {previous_inserts}, Current inserts: {inserts_count}")
                print(f"    Reused: {reused}/{max_possible} ({efficiency:.1f}%)")
        
        # Calculate maximum theoretical inserts that could be reused
        # For each transition, the maximum is the minimum of the two consecutive zip codes' insert counts
        theoretical_max = 0
        for i in range(1, len(assignments)):
            prev_inserts = len(assignments[i-1]['inserts'])
            curr_inserts = len(assignments[i]['inserts'])
            theoretical_max += min(prev_inserts, curr_inserts)
        
        # Calculate overall machine efficiency
        if transitions > 0:
            machine_efficiency = (machine_reused / theoretical_max) * 100 if theoretical_max > 0 else 0
            print(f"\n  Overall machine efficiency: {machine_reused}/{theoretical_max} ({machine_efficiency:.1f}%)")
        
        # Add to totals
        total_inserts += machine_inserts
        potential_total += theoretical_max
    
    # Calculate overall system efficiency
    overall_efficiency = (total_reused / potential_total) * 100 if potential_total > 0 else 0
    
    print("\n" + "=" * 50)
    print(f"Total ZIP codes processed: {total_zip_codes}")
    print(f"Total inserts across all machines: {total_inserts}")
    print(f"Total inserts reused: {total_reused}")
    print(f"Maximum theoretical reuse: {potential_total}")
    print(f"Overall system efficiency: {overall_efficiency:.1f}%")
    print("=" * 50)

if __name__ == "__main__":
    calculate_reuse_metrics() 