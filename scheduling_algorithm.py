"""
Fixed version of the scheduling algorithm that properly handles NaN values in CombinedStoreName
and ensures correct type conversion for zip codes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import os

def convert_to_serializable(obj):
    """
    Convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with NumPy types converted to Python types
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Special handling for segment and inserts_reused keys
        result = {}
        for k, v in obj.items():
            if k == 'segments' and v is None:
                result[k] = []
            elif k == 'segments' and isinstance(v, list):
                # Filter out empty or None segments and convert to strings
                result[k] = [str(s) for s in v if s is not None and str(s).strip()]
            elif k == 'inserts_reused' and not isinstance(v, (int, float)):
                # Ensure inserts_reused is a number
                try:
                    result[k] = int(v)
                except (ValueError, TypeError):
                    result[k] = 0
            else:
                result[k] = convert_to_serializable(v)
        return result
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, set):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

class SchedulingAlgorithm:
    """
    Algorithm to schedule production across multiple machines,
    maximizing insert continuity between consecutive zip codes.
    """
    
    def __init__(self, machines_count=3, pockets_per_machine=16):
        """
        Initialize the scheduling algorithm.
        
        Args:
            machines_count (int): Number of machines available
            pockets_per_machine (int): Number of pockets per machine
        """
        self.machines_count = machines_count
        self.pockets_per_machine = pockets_per_machine
        self.machine_names = [f"Machine {chr(65+i)}" for i in range(machines_count)]  # Machine A, B, C, etc.
        self.machine_states = {}
        self.zip_codes = []
        self.store_names_by_zip = {}
        self.inserts_by_zip = {}
        self.segments_by_zip = {}
        self.pieces_by_zip_insert = {}
        self.machine_specific_zips = {name: [] for name in self.machine_names}
        self.overlap_scores = {}
        self.inserts_used_count = 0
        self.zip_codes_df = None
        self.orders_df = None
        self.delivery_days = {}  # Store delivery days by zip code
        self.postal_scheduling = False
        self.postal_type_by_zip = {}
        self.mayhem_scheduling = False  # Whether to use mayhem mode (ignore postal/delivery constraints)
        
        self.initialize_machines()
        
    def initialize_machines(self):
        """Initialize the state of each machine with empty pockets."""
        for machine_id in range(1, self.machines_count + 1):
            self.machine_states[machine_id] = {
                'current_inserts': set(),  # Set of inserts currently in the machine
                'pocket_assignments': {},  # Mapping of pocket number to insert
                'scheduled_zip_codes': [],  # List of zip codes scheduled on this machine
                'last_zip_code': None,  # Last zip code scheduled on this machine
            }
    
    def load_data(self, zip_codes_df, orders_df):
        """
        Load and prepare the scheduling data.
        
        Args:
            zip_codes_df (DataFrame): DataFrame containing ZIP code data
            orders_df (DataFrame): DataFrame containing order data
        """
        self.zip_codes_df = zip_codes_df
        self.orders_df = orders_df
        
        # Convert zip codes to string for consistent comparison
        orders_df['ZipCode'] = orders_df['ZipCode'].astype(str)
        zip_codes_df['ZipCode'] = zip_codes_df['ZipCode'].astype(str)
        
        # Handle NaN in CombinedStoreName more appropriately
        # Clean up any NaN or blank CombinedStoreName values by combining the fields properly
        mask = orders_df['CombinedStoreName'].isna() | (orders_df['CombinedStoreName'] == '')
        for idx in orders_df[mask].index:
            advertiser = str(orders_df.loc[idx, 'AdvertiserAccount']).strip()
            store = str(orders_df.loc[idx, 'StoreName']).strip()
            version = str(orders_df.loc[idx, 'ZipRouteVersion']).strip()
            
            # Create a meaningful name using the available parts
            parts = [p for p in [advertiser, store, version] if p and p.lower() != 'nan']
            if parts:
                orders_df.loc[idx, 'CombinedStoreName'] = ' '.join(parts)
            else:
                # If all parts are NaN or empty, use a prefix with the index to make it unique
                orders_df.loc[idx, 'CombinedStoreName'] = f"Insert-{idx}"
        
        # Group by zip code
        grouped = orders_df.groupby('ZipCode')
        
        # Extract unique zip codes
        self.zip_codes = list(grouped.groups.keys())
        
        # Reset data structures
        self.store_names_by_zip = {}
        self.inserts_by_zip = {}
        self.segments_by_zip = {}  # Store segments for each zip code
        self.pieces_by_zip_insert = {}  # Store pieces count for each zip+insert
        self.delivery_days = {}  # Reset delivery days
        
        # Extract delivery days from zip_codes_df and ensure they're standardized to MON/TUE
        for _, row in zip_codes_df.iterrows():
            zip_code = str(row['ZipCode'])
            if 'mailday' in row and pd.notna(row['mailday']):
                mailday = str(row['mailday']).strip().upper()
                # Standardize day names
                if mailday in ['MON', 'MONDAY', 'M']:
                    self.delivery_days[zip_code] = 'MON'
                elif mailday in ['TUE', 'TUESDAY', 'T']:
                    self.delivery_days[zip_code] = 'TUE'
                else:
                    self.delivery_days[zip_code] = mailday
        
        # Process each zip code
        for zip_code in self.zip_codes:
            # Extract unique inserts for this zip
            zip_group = grouped.get_group(zip_code)
            
            # Get unique store names (inserts) for this zip, excluding empty ones
            inserts = set()
            for insert in zip_group['CombinedStoreName'].unique():
                if pd.notna(insert) and insert.strip() != '':
                    inserts.add(insert)
            
            self.inserts_by_zip[zip_code] = inserts
            
            # Get unique segments for this zip code
            segments = []
            for segment in zip_group['ZipRouteWithSegment'].unique():
                if pd.notna(segment) and str(segment).strip() != '':
                    segments.append(str(segment))
            
            # Debug print statement to check segments
            if len(segments) > 0:
                print(f"ZIP {zip_code} has {len(segments)} segments: {segments}")
            else:
                print(f"ZIP {zip_code} has no segments")
            
            self.segments_by_zip[zip_code] = segments
            
            # Store the total pieces for each insert in this zip
            for insert in inserts:
                insert_pieces = zip_group[zip_group['CombinedStoreName'] == insert]['Pieces'].sum()
                key = f"{zip_code}_{insert}"
                self.pieces_by_zip_insert[key] = insert_pieces
            
            # Calculate total inserts for this zip
            self.store_names_by_zip[zip_code] = len(inserts)
        
        # Reset machine-specific zips
        self.machine_specific_zips = {name: [] for name in self.machine_names}
        
        # Track the truckload assignments to ensure balance
        truckload_counts = {"CBAM1": 0, "CBAM2": 0, "CBAM3": 0}
        
        # Assign zip codes to specific machines based on truckload value
        for _, row in zip_codes_df.iterrows():
            zip_code = str(row['ZipCode'])
            if zip_code in self.zip_codes:
                truckload = str(row.get('truckload', '')).strip()
                
                # Extract the CBAM prefix if present
                truckload_prefix = None
                if truckload.startswith('CBAM1'):
                    truckload_prefix = "CBAM1"
                elif truckload.startswith('CBAM2'):
                    truckload_prefix = "CBAM2"
                elif truckload.startswith('CBAM3'):
                    truckload_prefix = "CBAM3"
                
                # Maintain daily delivery information
                delivery_day = row.get('mailday', '')
                if delivery_day:
                    if zip_code not in self.delivery_days:
                        # Standardize the delivery day format
                        mailday = str(delivery_day).strip().upper()
                        if mailday in ['MON', 'MONDAY', 'M']:
                            self.delivery_days[zip_code] = 'MON'
                        elif mailday in ['TUE', 'TUESDAY', 'T']:
                            self.delivery_days[zip_code] = 'TUE'
                        else:
                            self.delivery_days[zip_code] = mailday
                
                # Assign to machine based on truckload with balance check
                if truckload_prefix:
                    # Get corresponding machine name
                    machine_name = f"Machine {chr(64 + int(truckload_prefix[-1]))}"
                    
                    # Add to specific machine
                    self.machine_specific_zips[machine_name].append(zip_code)
                    truckload_counts[truckload_prefix] += 1
        
        # If any machine has significantly more ZIPs, try to balance
        avg_zips = sum(len(zips) for zips in self.machine_specific_zips.values()) / len(self.machine_names)
        
        # More aggressive balancing - if machine balance is off by more than 20% (was 30%), adjust
        for name, zips in self.machine_specific_zips.items():
            if len(zips) > avg_zips * 1.2:  # Machine has 20% more than average
                overflow_count = int(len(zips) - avg_zips)
                
                # Find machines with fewer than average ZIPs
                receiving_machines = [m for m in self.machine_names 
                                    if m != name and len(self.machine_specific_zips[m]) < avg_zips]
                
                if receiving_machines:
                    # Find ZIPs with fewer inserts to move (less disruption)
                    zip_insert_counts = [(z, self.store_names_by_zip.get(z, 0)) for z in zips]
                    zip_insert_counts.sort(key=lambda x: x[1])  # Sort by insert count (ascending)
                    
                    # Move more ZIPs to balance
                    for i in range(min(overflow_count, len(zip_insert_counts))):
                        target_machine = receiving_machines[i % len(receiving_machines)]
                        zip_to_move = zip_insert_counts[i][0]
                        
                        # Remove from source machine
                        self.machine_specific_zips[name].remove(zip_to_move)
                        
                        # Add to target machine
                        self.machine_specific_zips[target_machine].append(zip_to_move)
        
        # Calculate reuse scores
        self.calculate_overlap_scores()
        
        # Reset inserts_used_count
        self.inserts_used_count = 0
        
        # Ensure we have delivery_days initialized
        if not hasattr(self, 'delivery_days'):
            self.delivery_days = {}
    
    def calculate_overlap(self, zip_code1, zip_code2):
        """
        Calculate the overlap of inserts between two zip codes.
        
        Args:
            zip_code1 (str): First zip code
            zip_code2 (str): Second zip code
            
        Returns:
            int: Number of inserts that overlap between the two zip codes
        """
        if zip_code1 not in self.inserts_by_zip or zip_code2 not in self.inserts_by_zip:
            return 0
            
        inserts1 = self.inserts_by_zip[zip_code1]
        inserts2 = self.inserts_by_zip[zip_code2]
        
        # Calculate the intersection of the two sets
        overlap = inserts1.intersection(inserts2)
        return len(overlap)
    
    def calculate_overlap_scores(self):
        """
        Calculate and store overlap scores between all pairs of zip codes.
        This helps in determining which zip codes should be scheduled consecutively.
        """
        self.overlap_scores = {}
        
        # Calculate overlap for each pair of zip codes
        for zip1 in self.zip_codes:
            for zip2 in self.zip_codes:
                if zip1 != zip2:
                    overlap = self.calculate_overlap(zip1, zip2)
                    self.overlap_scores[(zip1, zip2)] = overlap
    
    def calculate_machine_compatibility(self, zip_code):
        """
        Calculate the compatibility score of a zip code with each machine.
        
        Args:
            zip_code (str): Zip code to check
            
        Returns:
            dict: Dictionary mapping machine_id to compatibility score
        """
        compatibility = {}
        
        for machine_id, state in self.machine_states.items():
            if state['last_zip_code'] is None:
                # If machine hasn't been used yet, assign a neutral score
                compatibility[machine_id] = 0
            else:
                # Calculate overlap with the last zip code on this machine
                last_zip = state['last_zip_code']
                overlap = self.calculate_overlap(last_zip, zip_code)
                compatibility[machine_id] = overlap
                
        return compatibility
    
    def assign_zip_to_machine(self, zip_code, machine_id):
        """
        Assign a zip code to a specific machine.
        
        Args:
            zip_code (str): Zip code to assign
            machine_id (int): Machine ID to assign to
            
        Returns:
            dict: Updated machine state
        """
        # Get the inserts for this zip code
        if zip_code not in self.inserts_by_zip:
            return self.machine_states[machine_id]
            
        zip_inserts = self.inserts_by_zip[zip_code]
        
        # Get the current state of the machine
        state = self.machine_states[machine_id]
        current_inserts = state['current_inserts']
        pocket_assignments = state['pocket_assignments']
        
        # Calculate which inserts need to be kept and which need to be added
        inserts_to_keep = current_inserts.intersection(zip_inserts)
        inserts_to_add = zip_inserts - current_inserts
        inserts_to_remove = current_inserts - zip_inserts
        
        # Update the pocket assignments
        # First, remove inserts that are no longer needed
        pockets_to_free = []
        for pocket_num, insert in list(pocket_assignments.items()):
            if insert in inserts_to_remove:
                pockets_to_free.append(pocket_num)
                del pocket_assignments[pocket_num]
                
        # Then, add new inserts to the freed pockets
        new_inserts = list(inserts_to_add)
        for i, pocket_num in enumerate(pockets_to_free):
            if i < len(new_inserts):
                pocket_assignments[pocket_num] = new_inserts[i]
                
        # If we need more pockets than were freed, use empty pockets
        if len(new_inserts) > len(pockets_to_free):
            remaining_inserts = new_inserts[len(pockets_to_free):]
            used_pockets = set(pocket_assignments.keys())
            empty_pockets = [p for p in range(1, self.pockets_per_machine + 1) 
                            if p not in used_pockets]
            
            for i, pocket_num in enumerate(empty_pockets):
                if i < len(remaining_inserts):
                    pocket_assignments[pocket_num] = remaining_inserts[i]
        
        # Update the machine state
        state['current_inserts'] = zip_inserts
        state['pocket_assignments'] = pocket_assignments
        state['scheduled_zip_codes'].append(zip_code)
        state['last_zip_code'] = zip_code
        
        self.machine_states[machine_id] = state
        return state
    
    def schedule_production(self, day=None):
        """
        Schedule production across all machines for a specific day,
        with a strong emphasis on maximizing insert continuity.
        
        Args:
            day (str, optional): Day to schedule (MON or TUE). If None, schedule both days.
            
        Returns:
            dict: Schedule results
        """
        # Reset machine states
        self.initialize_machines()
        
        # Filter zip codes by day if specified
        if day:
            zip_codes_to_schedule = self.zip_codes_df[self.zip_codes_df['mailday'] == day]['ZipCode'].astype(str).tolist()
        else:
            zip_codes_to_schedule = self.zip_codes_df['ZipCode'].astype(str).tolist()
        
        # Create a schedule
        schedule = {
            'machine_assignments': {},
            'total_inserts_reused': 0,
            'total_zip_codes': len(zip_codes_to_schedule),
            'schedule_date': datetime.now().strftime('%Y-%m-%d'),
            'day': day
        }
        
        # Initialize machine assignments in the schedule
        for machine_id in range(1, self.machines_count + 1):
            schedule['machine_assignments'][machine_id] = []
        
        # Filter to only zip codes that have inserts
        valid_zip_codes = []
        for zip_code in zip_codes_to_schedule:
            if zip_code in self.inserts_by_zip and len(self.inserts_by_zip[zip_code]) > 0:
                valid_zip_codes.append((zip_code, len(self.inserts_by_zip[zip_code])))
        
        # Sort zip codes by number of inserts (descending)
        valid_zip_codes.sort(key=lambda x: x[1], reverse=True)
        zip_codes_with_inserts = [z[0] for z in valid_zip_codes]
        
        # Start with the zip codes with the most inserts to seed the machines
        remaining_zip_codes = set(zip_codes_with_inserts)
        machine_current_zip = {}
        
        # Initialize each machine with one zip code (the ones with most inserts)
        for i in range(min(self.machines_count, len(zip_codes_with_inserts))):
            machine_id = i + 1
            zip_code = zip_codes_with_inserts[i]
            
            # Assign the zip code to the machine
            state = self.assign_zip_to_machine(zip_code, machine_id)
            
            schedule['machine_assignments'][machine_id].append({
                'zip_code': zip_code,
                'inserts': list(self.inserts_by_zip[zip_code]),  # Convert set to list
                'inserts_reused': 0,  # First assignment has no reuse
                'sequence': 1
            })
            
            # Mark as current zip for this machine and remove from remaining
            machine_current_zip[machine_id] = zip_code
            remaining_zip_codes.remove(zip_code)
        
        # Total inserts reused counter
        total_inserts_reused = 0
        
        # Process remaining zip codes by finding the best continuity for each machine
        while remaining_zip_codes:
            best_machine_id = None
            best_zip_code = None
            best_overlap = -1
            
            # Find the best machine-zip combination with maximum insert continuity
            for machine_id in range(1, self.machines_count + 1):
                if machine_id not in machine_current_zip:
                    continue
                    
                current_zip = machine_current_zip[machine_id]
                current_inserts = self.inserts_by_zip[current_zip]
                
                # Find best next zip code for this machine
                for zip_code in remaining_zip_codes:
                    zip_inserts = self.inserts_by_zip[zip_code]
                    overlap = len(current_inserts.intersection(zip_inserts))
                    
                    # Calculate percentage overlap relative to the total inserts in the zip
                    if len(zip_inserts) > 0:
                        overlap_percentage = overlap / len(zip_inserts)
                    else:
                        overlap_percentage = 0
                        
                    # Use overlap count and percentage as criteria
                    combined_score = overlap * (1 + overlap_percentage)
                    
                    if combined_score > best_overlap:
                        best_overlap = combined_score
                        best_machine_id = machine_id
                        best_zip_code = zip_code
            
            # If no good match found, assign to least loaded machine
            if best_machine_id is None or best_zip_code is None:
                machine_loads = [(m, len(schedule['machine_assignments'][m])) 
                                for m in range(1, self.machines_count + 1)]
                machine_loads.sort(key=lambda x: x[1])  # Sort by load (ascending)
                best_machine_id = machine_loads[0][0]
                best_zip_code = list(remaining_zip_codes)[0]
                best_overlap = 0
            
            # Assign the chosen zip code to the best machine
            state = self.assign_zip_to_machine(best_zip_code, best_machine_id)
            overlap_count = len(self.inserts_by_zip[machine_current_zip[best_machine_id]]
                               .intersection(self.inserts_by_zip[best_zip_code]))
            
            # Update the schedule
            sequence_num = len(schedule['machine_assignments'][best_machine_id]) + 1
            schedule['machine_assignments'][best_machine_id].append({
                'zip_code': best_zip_code,
                'inserts': list(self.inserts_by_zip[best_zip_code]),  # Convert set to list
                'inserts_reused': int(overlap_count),  # Convert to regular Python int to avoid numpy.int64
                'sequence': int(sequence_num)  # Convert to regular Python int to avoid numpy.int64
            })
            
            # Update tracking variables
            total_inserts_reused += overlap_count
            machine_current_zip[best_machine_id] = best_zip_code
            remaining_zip_codes.remove(best_zip_code)
        
        # Update total inserts reused in the schedule
        schedule['total_inserts_reused'] = int(total_inserts_reused)  # Convert to regular Python int
        
        # Add pocket changes information
        pocket_changes = self.generate_pocket_changes(schedule)
        schedule['pocket_changes'] = convert_to_serializable(pocket_changes)
        
        return schedule
    
    def _balance_machine_loads(self, schedule):
        """
        Balance the loads across machines if they are significantly imbalanced.
        
        Args:
            schedule (dict): Schedule to balance
        """
        # Calculate average assignments per machine
        total_assignments = sum(len(assignments) for assignments in schedule['machine_assignments'].values())
        avg_assignments = total_assignments / self.machines_count
        
        # Identify overloaded and underloaded machines
        overloaded = []
        underloaded = []
        
        for machine_id, assignments in schedule['machine_assignments'].items():
            load = len(assignments)
            if load < 0.7 * avg_assignments:
                underloaded.append((machine_id, load))
            elif load > 1.3 * avg_assignments:
                overloaded.append((machine_id, load))
        
        # Sort by load (ascending for underloaded, descending for overloaded)
        underloaded.sort(key=lambda x: x[1])
        overloaded.sort(key=lambda x: x[1], reverse=True)
        
        # Move assignments from overloaded to underloaded machines
        for under_id, under_load in underloaded:
            if not overloaded:
                break
                
            over_id, over_load = overloaded[0]
            
            # Calculate how many to move
            target = min(avg_assignments, under_load + (over_load - avg_assignments) / 2)
            to_move = int(target - under_load)
            
            if to_move > 0:
                # Move assignments (prioritize ones with fewer inserts to minimize disruption)
                assignments = schedule['machine_assignments'][over_id]
                assignments.sort(key=lambda x: len(x['inserts']))
                
                for i in range(min(to_move, len(assignments))):
                    # Move assignment
                    assignment = assignments[i]
                    schedule['machine_assignments'][under_id].append(assignment)
                
                # Remove moved assignments from overloaded machine
                schedule['machine_assignments'][over_id] = assignments[to_move:]
                
                # Update the overloaded machine's load
                overloaded[0] = (over_id, len(schedule['machine_assignments'][over_id]))
                
                # Re-sort overloaded list
                overloaded.sort(key=lambda x: x[1], reverse=True)
    
    def generate_pocket_changes(self, schedule):
        """
        Generate a list of pocket changes needed for each machine.
        
        Args:
            schedule (dict): Schedule generated by generate_schedule
            
        Returns:
            dict: Dictionary mapping machine_id to list of pocket changes
        """
        pocket_changes = {}
        
        # Process each machine in the schedule
        for machine_name, assignments in schedule.items():
            # Extract machine ID (numeric or from machine name)
            if isinstance(machine_name, int) or (isinstance(machine_name, str) and machine_name.isdigit()):
                machine_id = str(machine_name)
            else:
                # If machine_name is like "Machine A", extract the letter and convert to ID
                if len(machine_name) > 0 and "Machine " in machine_name:
                    letter = machine_name.replace("Machine ", "")[0]
                    machine_id = str(ord(letter) - 64)  # A -> 1, B -> 2, etc.
                else:
                    # Fallback if we can't determine the ID
                    machine_id = str(list(schedule.keys()).index(machine_name) + 1)
            
            pocket_changes[machine_id] = []
            current_pockets = {}  # pocket_number -> insert_name
            
            for i, assignment in enumerate(assignments):
                zip_code = assignment['zip_code']
                # Use "i+1" as sequence if not specified
                sequence = i + 1
                current_inserts = set(current_pockets.values())
                
                # Get inserts for this ZIP code
                if 'inserts' in assignment:
                    if isinstance(assignment['inserts'], set):
                        target_inserts = assignment['inserts']
                    else:
                        target_inserts = set(assignment['inserts'])
                else:
                    # Get inserts from the class data if not in the assignment
                    target_inserts = self.inserts_by_zip.get(zip_code, set())
                
                # For the first ZIP code, simply add inserts to empty pockets
                if i == 0:
                    for idx, insert in enumerate(target_inserts):
                        if idx < self.pockets_per_machine:
                            pocket_num = idx + 1
                            current_pockets[pocket_num] = insert
                            pocket_changes[machine_id].append({
                                'zip_code': zip_code,
                                'sequence': sequence,
                                'action': 'add',
                                'pocket': pocket_num,
                                'insert': insert
                            })
                    continue
                
                # For subsequent ZIP codes:
                # 1. First identify which inserts to keep (exact matches only)
                inserts_to_keep = current_inserts.intersection(target_inserts)
                pockets_to_keep = {}  # insert -> pocket_number
                
                # Find one pocket for each insert we want to keep
                for pocket_num, insert in current_pockets.items():
                    if insert in inserts_to_keep and insert not in pockets_to_keep:
                        pockets_to_keep[insert] = pocket_num
                
                # 2. Mark pockets for keeping
                changes = []
                new_pocket_state = {}
                for insert, pocket_num in pockets_to_keep.items():
                    changes.append({
                        'zip_code': zip_code,
                        'sequence': sequence,
                        'action': 'keep',
                        'pocket': pocket_num,
                        'insert': insert
                    })
                    new_pocket_state[pocket_num] = insert
                
                # 3. Mark remaining current pockets for removal
                for pocket_num, insert in current_pockets.items():
                    if pocket_num not in new_pocket_state:
                        changes.append({
                            'zip_code': zip_code,
                            'sequence': sequence,
                            'action': 'remove',
                            'pocket': pocket_num,
                            'insert': insert
                        })
                
                # 4. Add new inserts to available pockets
                inserts_to_add = target_inserts - inserts_to_keep
                available_pockets = [p for p in range(1, self.pockets_per_machine + 1)
                                   if p not in new_pocket_state]
                
                # Make sure we don't exceed available pockets
                add_count = min(len(inserts_to_add), len(available_pockets))
                inserts_to_add_list = list(inserts_to_add)[:add_count]
                
                for insert, pocket_num in zip(inserts_to_add_list, available_pockets[:add_count]):
                    changes.append({
                        'zip_code': zip_code,
                        'sequence': sequence,
                        'action': 'add',
                        'pocket': pocket_num,
                        'insert': insert
                    })
                    new_pocket_state[pocket_num] = insert
                
                # Update current pocket state for next iteration
                current_pockets = new_pocket_state
                pocket_changes[machine_id].extend(changes)
        
        return pocket_changes

    def generate_schedule(self):
        """
        Generate the schedule for each machine.
        
        Returns:
            dict: A dictionary with machine names as keys and schedules as values.
                  Each schedule is a list of dictionaries, where each dictionary contains:
                  - 'zip_code': The ZIP code
                  - 'inserts': List of inserts (store names) for this ZIP code
                  - 'machine': The machine name
                  - 'segments': List of ZIP route segments for this ZIP code
                  - 'pieces': Dictionary mapping insert names to piece counts
                  - 'mailday': Delivery day for this ZIP code
        """
        print("\nGenerating schedule...")
        
        # Initialize the schedule
        schedule = {machine: [] for machine in self.machine_names}
        
        # Reset inserts counter
        self.inserts_used_count = 0
        
        # Track which zip codes have been processed
        processed_zips = set()
        
        # Different grouping strategies based on scheduling method
        if self.mayhem_scheduling:
            # In mayhem mode, completely ignore delivery days and postal types
            # Just focus on maximizing insert continuity across all ZIPs
            mayhem_zips = []
            
            # Get all ZIPs with inserts
            for zip_code in self.zip_codes:
                # Skip ZIPs with no inserts
                if zip_code not in self.inserts_by_zip or not self.inserts_by_zip[zip_code]:
                    continue
                mayhem_zips.append(zip_code)
            
            print(f"Mayhem Mode ☠️ activated! Processing all {len(mayhem_zips)} ZIPs for maximum insert continuity")
            
            # Treat all ZIPs as a single group
            facility_groups = [
                ('MAYHEM', mayhem_zips)
            ]
        
        elif self.postal_scheduling:
            # Group ZIPs by postal facility type (SCF first, then DDU)
            scf_zips = []
            ddu_zips = []
            other_zips = []
            
            for zip_code in self.zip_codes:
                # Skip ZIPs with no inserts
                if zip_code not in self.inserts_by_zip or not self.inserts_by_zip[zip_code]:
                    continue
                
                postal_type = self.postal_type_by_zip.get(zip_code, '').upper()
                if postal_type == 'SCF':
                    scf_zips.append(zip_code)
                elif postal_type == 'DDU':
                    ddu_zips.append(zip_code)
                else:
                    other_zips.append(zip_code)
            
            print(f"ZIP counts by postal facility type: SCF={len(scf_zips)}, DDU={len(ddu_zips)}, OTHER={len(other_zips)}")
            
            # Process postal facility groups in priority order (SCF first, then DDU)
            facility_groups = [
                ('SCF', scf_zips),
                ('DDU', ddu_zips),
                ('OTHER', other_zips)
            ]
        else:
            # Original delivery day grouping
            # First pass: standardize all mailday values in self.delivery_days
            for zip_code, day in self.delivery_days.items():
                day_upper = str(day).upper().strip()
                if day_upper in ['MON', 'MONDAY', 'M']:
                    self.delivery_days[zip_code] = 'MON'
                elif day_upper in ['TUE', 'TUESDAY', 'T']:
                    self.delivery_days[zip_code] = 'TUE'
            
            # Group ZIPs by delivery day
            mon_zips = []
            tue_zips = []
            other_zips = []
            
            for zip_code in self.zip_codes:
                # Skip ZIPs with no inserts
                if zip_code not in self.inserts_by_zip or not self.inserts_by_zip[zip_code]:
                    continue
                    
                mailday = self.delivery_days.get(zip_code, '').upper()
                if mailday == 'MON':
                    mon_zips.append(zip_code)
                elif mailday == 'TUE':
                    tue_zips.append(zip_code)
                else:
                    # For ZIPs with no mailday, check if we can infer from ZIP code (even=MON, odd=TUE)
                    try:
                        zip_num = int(zip_code)
                        if zip_num % 2 == 0:
                            mon_zips.append(zip_code)
                            self.delivery_days[zip_code] = 'MON'
                        else:
                            tue_zips.append(zip_code)
                            self.delivery_days[zip_code] = 'TUE'
                    except (ValueError, TypeError):
                        other_zips.append(zip_code)
            
            print(f"ZIP counts by delivery day: MON={len(mon_zips)}, TUE={len(tue_zips)}, OTHER={len(other_zips)}")
            
            # Process delivery day groups in order (MON, TUE, OTHER)
            facility_groups = [
                ('MON', mon_zips),
                ('TUE', tue_zips),
                ('OTHER', other_zips)
            ]
        
        # Keep track of unique inserts across all machines
        unique_inserts_set = set()
        
        # Process each group, but distribute them across all machines
        for group_label, group_zips in facility_groups:
            print(f"Processing {group_label} ZIPs ({len(group_zips)} ZIPs)")
            
            # Skip if no ZIPs for this group
            if not group_zips:
                continue
                
            # Create a group-specific schedule for each machine
            group_schedule = {machine: [] for machine in self.machine_names}
            
            # Calculate inserts per zip and build overlap matrix for this group
            zip_inserts = {}
            overlap_matrix = {}
            
            for zip_code in group_zips:
                inserts = self.inserts_by_zip.get(zip_code, set())
                zip_inserts[zip_code] = inserts
                
                # Count unique inserts across all zips
                unique_inserts_set.update(inserts)
                
                # Calculate overlap with all other zips from the same group
                for other_zip in group_zips:
                    if zip_code != other_zip:
                        other_inserts = self.inserts_by_zip.get(other_zip, set())
                        overlap = len(inserts.intersection(other_inserts))
                        overlap_matrix[(zip_code, other_zip)] = overlap
            
            # Sort zips by number of inserts (descending) 
            # This helps ensure zips with more inserts are placed first for better continuity
            sorted_zips = sorted(zip_inserts.items(), key=lambda x: len(x[1]), reverse=True)
            
            # Keep track of which zips are assigned to each machine
            machine_zips = {machine: [] for machine in self.machine_names}
            machine_loads = {machine: 0 for machine in self.machine_names}
            
            # First, assign the zips with the most inserts to different machines
            # This gives each machine a good starting point
            for i, (zip_code, _) in enumerate(sorted_zips[:len(self.machine_names)]):
                machine = self.machine_names[i % len(self.machine_names)]
                machine_zips[machine].append(zip_code)
                machine_loads[machine] += 1
                processed_zips.add(zip_code)
            
            # Now assign the rest of the zips based on continuity and load balancing
            remaining_zips = [zip_code for zip_code, _ in sorted_zips if zip_code not in processed_zips]
            
            while remaining_zips:
                best_score = -1
                best_zip = None
                best_machine = None
                
                for zip_code in remaining_zips:
                    for machine in self.machine_names:
                        machine_zip_list = machine_zips[machine]
                        
                        # Skip empty machines - will be handled later
                        if not machine_zip_list:
                            continue
                        
                        # Calculate overlap with the last zip in this machine
                        last_zip = machine_zip_list[-1]
                        overlap = overlap_matrix.get((zip_code, last_zip), 0)
                        
                        # Calculate load factor (prefer less loaded machines)
                        load_ratio = 1.0 - (machine_loads[machine] / (sum(machine_loads.values()) + 1e-6))
                        
                        # Combined score: 60% overlap, 40% load balancing in standard modes
                        # For mayhem mode, we heavily prioritize overlap (90% overlap, 10% load balancing)
                        if self.mayhem_scheduling:
                            score = (overlap * 0.9) + (load_ratio * 10)
                        else:
                            score = (overlap * 0.6) + (load_ratio * 40)
                        
                        if score > best_score:
                            best_score = score
                            best_zip = zip_code
                            best_machine = machine
                
                # If we found a good match
                if best_zip and best_machine:
                    machine_zips[best_machine].append(best_zip)
                    machine_loads[best_machine] += 1
                    processed_zips.add(best_zip)
                    remaining_zips.remove(best_zip)
                else:
                    # If no machine has zips yet, or another issue occurred, 
                    # just assign to the least loaded machine
                    least_loaded = min(self.machine_names, key=lambda m: machine_loads[m])
                    if remaining_zips:
                        next_zip = remaining_zips[0]
                        machine_zips[least_loaded].append(next_zip)
                        machine_loads[least_loaded] += 1
                        processed_zips.add(next_zip)
                        remaining_zips.remove(next_zip)
                    else:
                        break
            
            # Now create the actual schedule assignments
            for machine, zip_list in machine_zips.items():
                for i, zip_code in enumerate(zip_list):
                    inserts = self.inserts_by_zip.get(zip_code, set())
                    segments = self.segments_by_zip.get(zip_code, [])
                    
                    # Calculate how many inserts are reused from previous zip
                    inserts_reused = 0
                    if i > 0:
                        prev_zip = zip_list[i-1]
                        prev_inserts = self.inserts_by_zip.get(prev_zip, set())
                        inserts_reused = len(inserts.intersection(prev_inserts))
                    
                    # Get delivery day
                    delivery_day = self.delivery_days.get(zip_code, "")
                    
                    # Get postal type
                    postal_type = ""
                    if self.postal_scheduling:
                        postal_type = self.postal_type_by_zip.get(zip_code, "DDU")
                        
                    # For mayhem mode, assign a pseudo delivery day for display purposes
                    if self.mayhem_scheduling and not delivery_day:
                        # Alternate between MON and TUE
                        delivery_day = "MON" if len(schedule[machine]) % 2 == 0 else "TUE"
                        self.delivery_days[zip_code] = delivery_day
                    
                    # Create the assignment record
                    assignment = {
                        "zip_code": zip_code,
                        "inserts": list(inserts),
                        "segments": segments,
                        "inserts_reused": inserts_reused
                    }
                    
                    # Add delivery day if available
                    if delivery_day:
                        assignment["mailday"] = delivery_day
                        
                    # Add postal type if available
                    if postal_type:
                        assignment["postal_type"] = postal_type
                        
                    # Add to the schedule
                    schedule[machine].append(assignment)
                    
                    # Count total inserts used
                    self.inserts_used_count += len(inserts)
        
        # Final machine load statistics
        for machine, assignments in schedule.items():
            if self.mayhem_scheduling:
                # In mayhem mode, just show total ZIPs
                print(f"  {machine}: {len(assignments)} total ZIPs")
            else:
                # Show breakdown by day in standard modes
                mon_count = sum(1 for a in assignments if a.get('mailday', '') == 'MON')
                tue_count = sum(1 for a in assignments if a.get('mailday', '') == 'TUE')
                other_count = len(assignments) - mon_count - tue_count
                print(f"  {machine}: {len(assignments)} total ZIPs - {mon_count} MON, {tue_count} TUE{', ' + str(other_count) + ' OTHER' if other_count > 0 else ''}")
        
        # Set scheduling method in output label
        if self.mayhem_scheduling:
            scheduling_method = "Mayhem ☠️ (No Logic)"
        elif self.postal_scheduling:
            scheduling_method = "Postal Facility (SCF/DDU)"
        else:
            scheduling_method = "Delivery Day"
            
        print(f"Schedule generated. Total inserts used: {len(unique_inserts_set)}")
        
        return schedule

    def output_schedule(self, schedule, output_file):
        """
        Output the schedule to a JSON file.
        
        Args:
            schedule (dict): The generated schedule
            output_file (str): Path to the output file
            
        Returns:
            dict: Statistics about the schedule
        """
        # Calculate statistics
        total_zips = 0
        total_inserts = set()  # Use a set to count unique inserts
        
        # Initialize the outputted schedule
        schedule_output = {
            'machines': [],
            'statistics': {},
            'date': datetime.now().strftime('%Y-%m-%d'),
            'machine_assignments': {},
            'pocket_changes': {}
        }
        
        # Process each machine
        for machine_name, assignments in schedule.items():
            # Extract machine ID (either numeric or from the machine name)
            if isinstance(machine_name, int) or machine_name.isdigit():
                machine_id = int(machine_name)
                machine_letter = chr(64 + machine_id)  # 1 -> A, 2 -> B, etc.
                formatted_name = f"Machine {machine_letter}"
            else:
                # If the machine_name is already like "Machine A"
                formatted_name = machine_name
                # Extract the ID for API consistency (A -> 1, B -> 2, etc.)
                if len(machine_name) > 0 and "Machine " in machine_name:
                    letter = machine_name.replace("Machine ", "")[0]
                    machine_id = ord(letter) - 64  # A -> 1, B -> 2, etc.
                else:
                    # Fallback if we can't determine the ID
                    machine_id = list(schedule.keys()).index(machine_name) + 1
            
            # Initialize machine data
            machine_data = {
                'name': formatted_name,
                'machine_id': machine_id,
                'zips': []
            }
            
            # Track assignments for this machine for the API
            schedule_output['machine_assignments'][str(machine_id)] = []
            
            # Process each ZIP code assignment
            for i, zip_data in enumerate(assignments):
                zip_code = zip_data['zip_code']
                inserts = list(self.inserts_by_zip.get(zip_code, set()))
                
                # Update totals
                total_zips += 1
                total_inserts.update(inserts)
                
                # Calculate how many inserts are reused from the previous ZIP
                inserts_reused = 0
                reused_inserts = []
                if i > 0:
                    prev_inserts = set(self.inserts_by_zip.get(assignments[i-1]['zip_code'], set()))
                    curr_inserts = set(inserts)
                    common_inserts = prev_inserts.intersection(curr_inserts)
                    inserts_reused = len(common_inserts)
                    reused_inserts = list(common_inserts)
                
                # Get segments for this ZIP
                segments = self.segments_by_zip.get(zip_code, [])
                
                # Get delivery day for this ZIP
                mailday = self.delivery_days.get(zip_code, "")
                
                # Get postal type (SCF/DDU) for this ZIP if postal scheduling is enabled
                postal_type = ""
                if self.postal_scheduling:
                    postal_type = self.postal_type_by_zip.get(zip_code, "DDU")
                
                # Create the ZIP data
                zip_output = {
                    'zip_code': zip_code,
                    'inserts': inserts,
                    'segments': segments,
                    'inserts_reused': inserts_reused,
                    'reused_inserts': reused_inserts,
                    'mailday': mailday
                }
                
                # Add postal type if available
                if postal_type:
                    zip_output['postal_type'] = postal_type
                
                # Add to machine data
                machine_data['zips'].append(zip_output)
                
                # Add to machine assignments for the API
                schedule_output['machine_assignments'][str(machine_id)].append({
                    'zip_code': zip_code,
                    'inserts': inserts,
                    'segments': segments
                })
            
            # Add machine data to the output
            schedule_output['machines'].append(machine_data)
        
        # Determine the scheduling method used
        if self.mayhem_scheduling:
            scheduling_method = "Mayhem ☠️ (No Logic)"
        elif self.postal_scheduling:
            scheduling_method = "Postal Facility (SCF/DDU)"
        else:
            scheduling_method = "Delivery Day"
        
        # Add statistics
        schedule_output['statistics'] = {
            'total_zips': total_zips,
            'total_inserts': len(total_inserts),
            'machines_count': self.machines_count,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'scheduling_method': scheduling_method
        }
        
        # Add postal facility statistics if using postal scheduling
        if self.postal_scheduling:
            # Count SCF and DDU ZIPs
            scf_count = sum(1 for machine in schedule_output['machines'] 
                           for zip_data in machine['zips'] 
                           if zip_data.get('postal_type', '') == 'SCF')
            ddu_count = sum(1 for machine in schedule_output['machines'] 
                           for zip_data in machine['zips'] 
                           if zip_data.get('postal_type', '') == 'DDU')
            
            # Add postal facility statistics
            schedule_output['statistics']['scf_count'] = scf_count
            schedule_output['statistics']['ddu_count'] = ddu_count
        
        # Add pocket changes to the output if they've been generated
        schedule_output['pocket_changes'] = self.generate_pocket_changes(schedule)
        
        # Convert to JSON-serializable format and write to file
        with open(output_file, 'w') as f:
            json.dump(convert_to_serializable(schedule_output), f, indent=2)
        
        # Return statistics for reference
        return schedule_output['statistics']

def load_and_prepare_data(zip_file_path, orders_file_path):
    """
    Load and prepare data from the input files.
    
    Args:
        zip_file_path (str): Path to the Zips by Address File Group Excel file
        orders_file_path (str): Path to the Insert Orders CSV file
        
    Returns:
        tuple: (zip_codes_df, orders_df)
    """
    # Load zip codes
    zip_codes_df = pd.read_excel(zip_file_path)
    
    # Rename 'zip' column to 'ZipCode' for consistency
    if 'zip' in zip_codes_df.columns:
        zip_codes_df = zip_codes_df.rename(columns={'zip': 'ZipCode'})
    
    # Load orders
    orders_df = pd.read_csv(orders_file_path)
    
    # Filter for CBA LONG ISLAND
    cba_orders = orders_df[orders_df['DistributorName'] == 'CBA LONG ISLAND'].copy()
    
    # Create combined store name
    cba_orders['CombinedStoreName'] = cba_orders['AdvertiserAccount'] + ' ' + \
                                     cba_orders['StoreName'] + ' ' + \
                                     cba_orders['ZipRouteVersion']
    
    # Extract Base ZIP code (first 5 digits) for matching with zip_codes_df
    cba_orders['BaseZipCode'] = cba_orders['ZipRoute'].astype(str).str[:5]
    
    # Keep the full ZipRoute as well (includes segment like 11040A)
    cba_orders['ZipRouteWithSegment'] = cba_orders['ZipRoute']
    
    # Ensure Pieces is numeric
    cba_orders['Pieces'] = pd.to_numeric(cba_orders['Pieces'], errors='coerce').fillna(0).astype(int)
    
    # For backward compatibility (until we update all code references)
    cba_orders['ZipCode'] = cba_orders['BaseZipCode']
    
    return zip_codes_df, cba_orders

def schedule_machines(zip_file_path, orders_file_path, output_file_path):
    """
    Main function to schedule machines with ZIP codes and inserts.
    
    Args:
        zip_file_path (str): Path to the ZIP code data CSV
        orders_file_path (str): Path to the orders data CSV
        output_file_path (str): Path to write the output JSON
        
    Returns:
        dict: Statistics about the generated schedule
    """
    # Load and prepare data
    zip_codes_df, orders_df = load_and_prepare_data(zip_file_path, orders_file_path)
    
    # Create scheduler instance
    scheduler = SchedulingAlgorithm()
    
    # Load data into the scheduler
    scheduler.load_data(zip_codes_df, orders_df)
    
    # Generate schedule
    schedule = scheduler.generate_schedule()
    
    # Output schedule to file
    stats = scheduler.output_schedule(schedule, output_file_path)
    
    return stats

def run_scheduling(zip_file_path, orders_file_path, day=None, output_file_path=None, postal_scheduling=False, mayhem_scheduling=False):
    """
    Wrapper function for backward compatibility.
    Run the scheduling algorithm and return the schedule.
    
    Args:
        zip_file_path (str): Path to the ZIP code data file
        orders_file_path (str): Path to the orders data file
        day (str, optional): Day filter (MON, TUE, or None for all days)
        output_file_path (str, optional): Path to write the output. If None, a default path is used.
        postal_scheduling (bool, optional): Whether to schedule based on postal facility type (SCF/DDU)
        mayhem_scheduling (bool, optional): Whether to use mayhem mode (ignore postal/delivery constraints)
        
    Returns:
        dict: The generated schedule
    """
    # If no output path specified, create a default one
    if output_file_path is None:
        output_file_path = os.path.join(
            os.path.dirname(os.path.abspath(orders_file_path)), 
            'schedule_result.json'
        )
    
    # Load and prepare data
    zip_codes_df, orders_df = load_and_prepare_data(zip_file_path, orders_file_path)
    
    # Standardize day parameter
    if day:
        day = day.upper().strip()
        if day in ['MON', 'MONDAY', 'M']:
            day = 'MON'
        elif day in ['TUE', 'TUESDAY', 'T']:
            day = 'TUE'
    
    # Create a copy of zip_codes_df for filtering
    filtered_zip_codes_df = zip_codes_df.copy()
    
    # Filter by day if specified and not in mayhem mode
    if day and not mayhem_scheduling:
        # First, standardize the mailday values in the dataframe
        filtered_zip_codes_df['mailday'] = filtered_zip_codes_df['mailday'].apply(
            lambda x: 'MON' if str(x).upper().strip() in ['MON', 'MONDAY', 'M'] 
            else ('TUE' if str(x).upper().strip() in ['TUE', 'TUESDAY', 'T'] else str(x))
        )
        
        # Then filter to only include ZIPs with the specified mailday
        filtered_zip_codes_df = filtered_zip_codes_df[filtered_zip_codes_df['mailday'] == day]
        
        print(f"Filtering for delivery day: {day}")
        print(f"Kept {len(filtered_zip_codes_df)} ZIP codes out of {len(zip_codes_df)}")
    
    # If postal scheduling is enabled and not in mayhem mode, sort by postal facility type
    if postal_scheduling and not mayhem_scheduling:
        print("Using Postal Facility (SCF/DDU) scheduling mode")
        
        # Ensure ratedesc column exists and has valid values
        if 'ratedesc' not in filtered_zip_codes_df.columns:
            print("WARNING: 'ratedesc' column not found in ZIP code data")
        else:
            # Make sure ratedesc values are standardized
            filtered_zip_codes_df['ratedesc'] = filtered_zip_codes_df['ratedesc'].fillna('DDU')
            filtered_zip_codes_df['ratedesc'] = filtered_zip_codes_df['ratedesc'].apply(
                lambda x: str(x).strip().upper() if pd.notna(x) else 'DDU'
            )
            
            # Create a sorting key column (0 for SCF, 1 for DDU)
            filtered_zip_codes_df['postal_sort'] = filtered_zip_codes_df['ratedesc'].apply(
                lambda x: 0 if x == 'SCF' else 1
            )
            
            # Sort by postal_sort to prioritize SCF before DDU
            filtered_zip_codes_df = filtered_zip_codes_df.sort_values(by=['postal_sort', 'ZipCode'])
            
            print(f"Scheduling priority: {len(filtered_zip_codes_df[filtered_zip_codes_df['ratedesc'] == 'SCF'])} SCF ZIPs, "
                  f"then {len(filtered_zip_codes_df[filtered_zip_codes_df['ratedesc'] == 'DDU'])} DDU ZIPs")
            
    # If mayhem scheduling is enabled, print a message
    if mayhem_scheduling:
        print("Using Mayhem ☠️ scheduling mode (ignoring delivery days & postal types)")
    
    # Create scheduler instance
    scheduler = SchedulingAlgorithm()
    
    # Load data into the scheduler
    scheduler.load_data(filtered_zip_codes_df, orders_df)
    
    # Set scheduling flags
    scheduler.postal_scheduling = postal_scheduling
    scheduler.mayhem_scheduling = mayhem_scheduling
    
    # Store postal type (SCF/DDU) for each ZIP code if postal scheduling is enabled
    if postal_scheduling:
        scheduler.postal_type_by_zip = {}
        for _, row in filtered_zip_codes_df.iterrows():
            zip_code = str(row['ZipCode']).strip()
            if 'ratedesc' in row and pd.notna(row['ratedesc']):
                scheduler.postal_type_by_zip[zip_code] = str(row['ratedesc']).strip().upper()
            else:
                scheduler.postal_type_by_zip[zip_code] = 'DDU'  # Default to DDU if not specified
    
    # Generate schedule
    schedule = scheduler.generate_schedule()
    
    # Output schedule to file
    stats = scheduler.output_schedule(schedule, output_file_path)
    
    # For backward compatibility, return the schedule itself
    return schedule
