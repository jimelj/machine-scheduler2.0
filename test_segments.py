#!/usr/bin/env python3
import os
import pandas as pd
import json
from scheduling_algorithm import load_and_prepare_data, SchedulingAlgorithm

def test_segments():
    # Get paths from the app config
    upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    session_file = os.path.join(upload_folder, 'session.json')
    
    if not os.path.exists(session_file):
        print("Session file not found. Please upload files first.")
        return
        
    # Load the session data
    with open(session_file, 'r') as f:
        session_data = json.load(f)
    
    # Get the file paths
    zips_path = session_data.get('zips_path')
    orders_path = session_data.get('insert_orders_path')
    
    if not zips_path or not orders_path:
        print("File paths not found in session data.")
        return
    
    print(f"ZIP file: {zips_path}")
    print(f"Orders file: {orders_path}")
    
    # Load and prepare the data
    try:
        zip_codes_df, orders_df = load_and_prepare_data(zips_path, orders_path)
        
        print("\nOrders data - first few rows:")
        print(orders_df.head())
        
        print("\nChecking ZipRouteWithSegment column:")
        if 'ZipRouteWithSegment' in orders_df.columns:
            print(f"Total segments: {len(orders_df['ZipRouteWithSegment'].unique())}")
            print(f"Sample segments: {orders_df['ZipRouteWithSegment'].unique()[:10]}")
        else:
            print("ZipRouteWithSegment column not found!")
            print(f"Available columns: {orders_df.columns.tolist()}")
        
        # Create scheduler and check segments
        scheduler = SchedulingAlgorithm()
        scheduler.load_data(zip_codes_df, orders_df)
        
        print("\nChecking segments_by_zip in scheduler:")
        sample_zips = list(scheduler.segments_by_zip.keys())[:5]
        for zip_code in sample_zips:
            segments = scheduler.segments_by_zip.get(zip_code, [])
            print(f"ZIP {zip_code}: {len(segments)} segments - {segments}")
        
        # Check segments in the schedule
        schedule = scheduler.generate_schedule()
        
        print("\nChecking segments in the generated schedule:")
        for machine, zip_list in schedule.items():
            if not zip_list:
                continue
                
            print(f"\nMachine {machine}:")
            for idx, zip_entry in enumerate(zip_list[:3]):  # Just show first 3 for brevity
                segments = zip_entry.get('segments', [])
                print(f"  ZIP {zip_entry['zip_code']}: {len(segments)} segments - {segments}")
        
        # Verify segments in the output JSON
        output_path = os.path.join(upload_folder, 'schedule_result.json')
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                output_data = json.load(f)
                
            print("\nChecking segments in the output JSON file:")
            for machine_data in output_data.get('machines', [])[:1]:  # Just check first machine
                print(f"\nMachine {machine_data['name']}:")
                for idx, zip_data in enumerate(machine_data.get('zips', [])[:3]):  # Just show first 3
                    segments = zip_data.get('segments', [])
                    print(f"  ZIP {zip_data['zip_code']}: {len(segments)} segments - {segments}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_segments() 