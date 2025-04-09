"""
Flask application for Production Scheduler

This module implements the Flask web application for scheduling production
across 3 machines with 16 pockets each, maximizing insert continuity.
"""

import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import tempfile
import uuid

from database_schema import db, ZipCode, Advertiser, Order, Machine, MachinePocket, ScheduleItem, PocketAssignment, ScheduleRun
from scheduling_algorithm import SchedulingAlgorithm, load_and_prepare_data, schedule_machines, run_scheduling

# Create Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'production-scheduler-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///production_scheduler.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db.init_app(app)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the home page."""
    current_data = None
    try:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'session.json'), 'r') as f:
            session_data = json.load(f)
            if 'insert_orders_path' in session_data:
                # Extract just the filename from the path
                current_data = os.path.basename(session_data['insert_orders_path'])
    except FileNotFoundError:
        pass
    
    return render_template('index.html', current_data=current_data)

@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    """Handle file uploads."""
    if request.method == 'POST':
        # Check if the post request has the file parts
        if 'insert_orders' not in request.files:
            flash('No insert orders file part')
            return redirect(request.url)
            
        insert_orders_file = request.files['insert_orders']
        
        # If user does not select file, browser also submits an empty part without filename
        if insert_orders_file.filename == '':
            flash('No insert orders file selected')
            return redirect(request.url)
            
        if insert_orders_file and allowed_file(insert_orders_file.filename):
            filename = secure_filename(insert_orders_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            insert_orders_file.save(file_path)
            
            # Store the file path in the session
            session_data = {
                'insert_orders_path': file_path,
                'zips_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload', 'Zips by Address File Group.xlsx')
            }
            
            with open(os.path.join(app.config['UPLOAD_FOLDER'], 'session.json'), 'w') as f:
                json.dump(session_data, f)
                
            flash('File successfully uploaded')
            return redirect(url_for('schedule'))
    
    return render_template('upload.html')

@app.route('/schedule', methods=['GET', 'POST'])
def schedule():
    """Generate and display the production schedule."""
    # Load session data
    try:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'session.json'), 'r') as f:
            session_data = json.load(f)
    except FileNotFoundError:
        flash('Please upload insert orders file first')
        return redirect(url_for('upload_files'))
    
    if request.method == 'POST':
        # Get scheduling parameters
        day = request.form.get('day', None)
        if day == 'all':
            day = None
        
        try:
            # Run the scheduling algorithm
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'schedule_result.json')
            
            # Use run_scheduling to handle the day filter correctly
            schedule_result = run_scheduling(
                session_data['zips_path'],
                session_data['insert_orders_path'],
                day=day,
                output_file_path=output_path
            )
            
            # Read the generated schedule
            with open(output_path, 'r') as f:
                schedule_result = json.load(f)
            
            flash('Schedule generated successfully')
            return redirect(url_for('view_schedule'))
        except Exception as e:
            flash(f'Error generating schedule: {str(e)}')
            return redirect(request.url)
    
    return render_template('schedule_form.html')

@app.route('/view_schedule')
def view_schedule():
    """View the generated schedule."""
    try:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'schedule_result.json'), 'r') as f:
            schedule_result = json.load(f)
            
        # Ensure statistics exists and contains total_inserts
        if 'statistics' not in schedule_result:
            schedule_result['statistics'] = {}
            
        if 'total_inserts' not in schedule_result['statistics'] or not schedule_result['statistics']['total_inserts']:
            # Calculate total unique inserts across all machines if not already in the statistics
            all_inserts = set()
            for machine in schedule_result.get('machines', []):
                for zip_data in machine.get('zips', []):
                    all_inserts.update(zip_data.get('inserts', []))
            
            schedule_result['statistics']['total_inserts'] = len(all_inserts)
            
            # Save updated schedule back to file with total_inserts
            with open(os.path.join(app.config['UPLOAD_FOLDER'], 'schedule_result.json'), 'w') as f:
                json.dump(schedule_result, f, indent=2)
                
        # Calculate total inserts for each machine
        for machine in schedule_result.get('machines', []):
            machine_inserts = set()
            for zip_data in machine.get('zips', []):
                if 'inserts' in zip_data:
                    machine_inserts.update(zip_data['inserts'])
            machine['total_inserts'] = len(machine_inserts)
    
    except FileNotFoundError:
        flash('No schedule has been generated yet')
        return redirect(url_for('schedule'))
    
    return render_template('view_schedule.html', schedule=schedule_result)

@app.route('/machine/<int:machine_id>')
def machine_detail(machine_id):
    """View detailed information for a specific machine."""
    try:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'schedule_result.json'), 'r') as f:
            schedule_result = json.load(f)
    except FileNotFoundError:
        flash('No schedule has been generated yet')
        return redirect(url_for('schedule'))
    
    # Get machine data
    machine_index = machine_id - 1 # Convert from 1-based to 0-based index
    
    if machine_index < 0 or machine_index >= len(schedule_result['machines']):
        flash('Invalid machine ID')
        return redirect(url_for('view_schedule'))
    
    machine_data = schedule_result['machines'][machine_index]
    
    # Debug print statements for segments
    print(f"\nMachine {machine_data['name']} has {len(machine_data['zips'])} ZIPs")
    
    # Debug to check if inserts are present
    zip_with_inserts = 0
    for zip_data in machine_data['zips']:
        if 'inserts' in zip_data and zip_data['inserts']:
            zip_with_inserts += 1
            first_few = list(zip_data['inserts'])[:3]
            print(f"ZIP {zip_data['zip_code']} has {len(zip_data['inserts'])} inserts, first few: {first_few}")
        else:
            print(f"WARNING: ZIP {zip_data['zip_code']} has NO inserts!")
    
    print(f"Found {zip_with_inserts} ZIPs with inserts out of {len(machine_data['zips'])} total")
    
    for i, zip_data in enumerate(machine_data['zips']):
        segments = zip_data.get('segments', [])
        inserts_reused = zip_data.get('inserts_reused', 0)
        print(f"ZIP {zip_data['zip_code']}: {len(segments)} segments - {segments}, inserts_reused: {inserts_reused}")
    
    # Calculate total inserts for this machine
    total_machine_inserts = 0
    all_inserts_set = set()
    
    # Calculate reuse information for each zip code
    for i, zip_data in enumerate(machine_data['zips']):
        # Count inserts for this ZIP
        if 'inserts' in zip_data and zip_data['inserts']:
            current_inserts = set(zip_data['inserts'])
            all_inserts_set.update(current_inserts)
            
        # Make sure each zip has a consistent data structure
        if i > 0:
            prev_inserts = set(machine_data['zips'][i-1]['inserts'])
            curr_inserts = set(zip_data['inserts'])
            common_inserts = prev_inserts.intersection(curr_inserts)
            zip_data['reused_inserts'] = list(common_inserts)
            
            # If inserts_reused is not already in the data (from schedule output), calculate it
            if 'inserts_reused' not in zip_data:
                zip_data['inserts_reused'] = len(common_inserts)
                print(f"Calculated inserts_reused for {zip_data['zip_code']}: {len(common_inserts)}")
        else:
            zip_data['reused_inserts'] = []
            
            # First ZIP has 0 reused inserts
            if 'inserts_reused' not in zip_data:
                zip_data['inserts_reused'] = 0
                print(f"Set inserts_reused=0 for first zip {zip_data['zip_code']}")
        
        # Ensure mailday is present
        if 'mailday' not in zip_data:
            zip_data['mailday'] = ""
            
        # Ensure segments is present and properly formatted
        if 'segments' not in zip_data:
            zip_data['segments'] = []
        elif zip_data['segments'] is None:
            zip_data['segments'] = []
        else:
            # Make sure segments are strings and filter out empty ones
            zip_data['segments'] = [str(s) for s in zip_data['segments'] if s]
            print(f"Processed segments for {zip_data['zip_code']}: {zip_data['segments']}")
    
    # Final validation before sending to template
    for zip_data in machine_data['zips']:
        print(f"Final check - ZIP {zip_data['zip_code']}: segments={zip_data.get('segments', [])}, inserts_reused={zip_data.get('inserts_reused', 0)}")
        # Ensure these values are JSON serializable
        if not isinstance(zip_data.get('segments', []), list):
            zip_data['segments'] = []
        if not isinstance(zip_data.get('inserts_reused', 0), (int, float)):
            zip_data['inserts_reused'] = 0
            
        # Ensure mailday is properly set
        if 'mailday' not in zip_data or not zip_data['mailday'] or zip_data['mailday'] == '-':
            # Try to determine mailday from the ZIP code segment
            # Many segments follow a pattern where they start with the ZIP code
            segments = zip_data.get('segments', [])
            if segments:
                # Check if any segment contains "MON" or "TUE"
                for segment in segments:
                    if "MON" in segment.upper():
                        zip_data['mailday'] = "MON"
                        break
                    elif "TUE" in segment.upper():
                        zip_data['mailday'] = "TUE"
                        break
            
            # If still not set, use a default based on ZIP code
            # This is a fallback mechanism - even numbered ZIPs go to Monday, odd to Tuesday
            if not zip_data.get('mailday'):
                try:
                    zip_num = int(zip_data['zip_code'])
                    zip_data['mailday'] = "MON" if zip_num % 2 == 0 else "TUE"
                except (ValueError, TypeError):
                    # If conversion fails, set a default
                    zip_data['mailday'] = "MON"
        
        # Standardize mailday format
        if zip_data.get('mailday'):
            mailday = zip_data['mailday'].upper().strip()
            if mailday in ['MON', 'MONDAY', 'M']:
                zip_data['mailday'] = 'MON'
            elif mailday in ['TUE', 'TUESDAY', 'T']:
                zip_data['mailday'] = 'TUE'
        else:
            # Make sure we always have a mailday value
            zip_data['mailday'] = 'MON'
    
    # Set the total inserts count for this machine (unique inserts)
    total_machine_inserts = len(all_inserts_set)
    
    # Generate empty placeholder for pocket changes to avoid JSON error
    pocket_changes = []
    
    return render_template(
        'machine_detail.html',
        machine_id=machine_id,
        machine_name=machine_data['name'],
        assignments=machine_data['zips'],
        total_inserts=total_machine_inserts,
        pocket_changes=pocket_changes
    )

@app.route('/api/schedule')
def api_schedule():
    """API endpoint to get the schedule data."""
    try:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'schedule_result.json'), 'r') as f:
            schedule_result = json.load(f)
        return jsonify(schedule_result)
    except FileNotFoundError:
        return jsonify({'error': 'No schedule has been generated yet'}), 404

@app.route('/api/machine/<int:machine_id>')
def api_machine(machine_id):
    """API endpoint to get data for a specific machine."""
    try:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'schedule_result.json'), 'r') as f:
            schedule_result = json.load(f)
        
        # Get machine assignments
        machine_assignments = schedule_result['machine_assignments'].get(str(machine_id), [])
        
        # Get pocket changes
        pocket_changes = schedule_result['pocket_changes'].get(str(machine_id), [])
        
        return jsonify({
            'machine_id': machine_id,
            'assignments': machine_assignments,
            'pocket_changes': pocket_changes
        })
    except FileNotFoundError:
        return jsonify({'error': 'No schedule has been generated yet'}), 404

@app.route('/initialize_db')
def initialize_db():
    """Initialize the database with default data."""
    with app.app_context():
        db.create_all()
        
        # Add machines if they don't exist
        for i in range(1, 4):
            machine = Machine.query.filter_by(name=f'Machine {i}').first()
            if not machine:
                machine = Machine(name=f'Machine {i}', pocket_count=16)
                db.session.add(machine)
                
                # Add pockets for this machine
                for j in range(1, 17):
                    pocket = MachinePocket(machine=machine, pocket_number=j)
                    db.session.add(pocket)
        
        db.session.commit()
        
        flash('Database initialized successfully')
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/debug/<int:machine_id>')
def debug_machine(machine_id):
    """Debug view for machine data."""
    try:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'schedule_result.json'), 'r') as f:
            schedule_result = json.load(f)
    except FileNotFoundError:
        flash('No schedule has been generated yet')
        return redirect(url_for('schedule'))
    
    # Get machine data
    machine_index = machine_id - 1 # Convert from 1-based to 0-based index
    
    if machine_index < 0 or machine_index >= len(schedule_result['machines']):
        flash('Invalid machine ID')
        return redirect(url_for('view_schedule'))
    
    machine_data = schedule_result['machines'][machine_index]
    
    # Debug print statements
    print(f"\nDEBUG - Machine {machine_data['name']} has {len(machine_data['zips'])} ZIPs")
    
    # Process data for template
    for zip_data in machine_data['zips']:
        # Process segments
        if 'segments' not in zip_data or zip_data['segments'] is None:
            zip_data['segments'] = []
        
        # Process inserts_reused
        if 'inserts_reused' not in zip_data:
            # If previous zip exists, calculate value
            idx = machine_data['zips'].index(zip_data)
            if idx > 0:
                prev_zip = machine_data['zips'][idx-1]
                prev_inserts = set(prev_zip['inserts'])
                curr_inserts = set(zip_data['inserts'])
                zip_data['inserts_reused'] = len(prev_inserts.intersection(curr_inserts))
            else:
                zip_data['inserts_reused'] = 0
        
        # Debug print
        print(f"ZIP {zip_data['zip_code']}: segments={zip_data['segments']}, inserts_reused={zip_data['inserts_reused']}")
    
    return render_template(
        'debug.html',
        machine_id=machine_id,
        machine_name=machine_data['name'],
        assignments=machine_data['zips']
    )

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('cba-logo.svg')

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500

@app.route('/operator_report/<int:machine_id>')
def operator_report(machine_id):
    """Generate a printable operator report for a specific machine."""
    try:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'schedule_result.json'), 'r') as f:
            schedule_result = json.load(f)
    except FileNotFoundError:
        flash('No schedule has been generated yet')
        return redirect(url_for('schedule'))
    
    # Get machine data
    machine_index = machine_id - 1  # Convert from 1-based to 0-based index
    
    if machine_index < 0 or machine_index >= len(schedule_result['machines']):
        flash('Invalid machine ID')
        return redirect(url_for('view_schedule'))
    
    machine_data = schedule_result['machines'][machine_index]
    
    # Calculate total inserts for this machine
    all_inserts_set = set()
    
    # Calculate reuse information for each zip code
    for i, zip_data in enumerate(machine_data['zips']):
        # Count inserts for this ZIP
        if 'inserts' in zip_data and zip_data['inserts']:
            current_inserts = set(zip_data['inserts'])
            all_inserts_set.update(current_inserts)
            
        # Make sure each zip has a consistent data structure
        if i > 0:
            prev_inserts = set(machine_data['zips'][i-1]['inserts'])
            curr_inserts = set(zip_data['inserts'])
            common_inserts = prev_inserts.intersection(curr_inserts)
            zip_data['reused_inserts'] = list(common_inserts)
            
            # If inserts_reused is not already in the data (from schedule output), calculate it
            if 'inserts_reused' not in zip_data:
                zip_data['inserts_reused'] = len(common_inserts)
        else:
            zip_data['reused_inserts'] = []
            
            # First ZIP has 0 reused inserts
            if 'inserts_reused' not in zip_data:
                zip_data['inserts_reused'] = 0
        
        # Ensure mailday is present
        if 'mailday' not in zip_data:
            zip_data['mailday'] = ""
            
        # Ensure segments is present and properly formatted
        if 'segments' not in zip_data:
            zip_data['segments'] = []
        elif zip_data['segments'] is None:
            zip_data['segments'] = []
        else:
            # Make sure segments are strings and filter out empty ones
            zip_data['segments'] = [str(s) for s in zip_data['segments'] if s]
    
    # Final validation before sending to template
    for zip_data in machine_data['zips']:
        # Ensure these values are JSON serializable
        if not isinstance(zip_data.get('segments', []), list):
            zip_data['segments'] = []
        if not isinstance(zip_data.get('inserts_reused', 0), (int, float)):
            zip_data['inserts_reused'] = 0
            
        # Ensure mailday is properly set
        if 'mailday' not in zip_data or not zip_data['mailday'] or zip_data['mailday'] == '-':
            # Try to determine mailday from the ZIP code segment
            # Many segments follow a pattern where they start with the ZIP code
            segments = zip_data.get('segments', [])
            if segments:
                # Check if any segment contains "MON" or "TUE"
                for segment in segments:
                    if "MON" in segment.upper():
                        zip_data['mailday'] = "MON"
                        break
                    elif "TUE" in segment.upper():
                        zip_data['mailday'] = "TUE"
                        break
            
            # If still not set, use a default based on ZIP code
            # This is a fallback mechanism - even numbered ZIPs go to Monday, odd to Tuesday
            if not zip_data.get('mailday'):
                try:
                    zip_num = int(zip_data['zip_code'])
                    zip_data['mailday'] = "MON" if zip_num % 2 == 0 else "TUE"
                except (ValueError, TypeError):
                    # If conversion fails, set a default
                    zip_data['mailday'] = "MON"
        
        # Standardize mailday format
        if zip_data.get('mailday'):
            mailday = zip_data['mailday'].upper().strip()
            if mailday in ['MON', 'MONDAY', 'M']:
                zip_data['mailday'] = 'MON'
            elif mailday in ['TUE', 'TUESDAY', 'T']:
                zip_data['mailday'] = 'TUE'
        else:
            # Make sure we always have a mailday value
            zip_data['mailday'] = 'MON'
    
    # Set the total inserts count for this machine (unique inserts)
    total_machine_inserts = len(all_inserts_set)
    
    # Pre-calculate all pocket changes for each ZIP
    # This version is designed for a printable report
    pocket_changes = {}
    current_pockets = {}  # pocket_number -> insert_name
    
    for i, assignment in enumerate(machine_data['zips']):
        zip_code = assignment['zip_code']
        sequence = i + 1
        current_inserts = set(current_pockets.values())
        target_inserts = set(assignment['inserts'])
        
        pocket_changes[zip_code] = []
        
        # For the first ZIP code, simply add inserts to empty pockets
        if i == 0:
            for idx, insert in enumerate(target_inserts):
                if idx < 16:  # Max 16 pockets
                    pocket_num = idx + 1
                    current_pockets[pocket_num] = insert
                    pocket_changes[zip_code].append({
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
        pockets_to_keep = {}
        
        # Find one pocket for each insert we want to keep
        for pocket_num, insert in current_pockets.items():
            if insert in inserts_to_keep and insert not in pockets_to_keep:
                pockets_to_keep[insert] = pocket_num
        
        # 2. Mark pockets for keeping
        new_pocket_state = {}
        for insert, pocket_num in pockets_to_keep.items():
            pocket_changes[zip_code].append({
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
                pocket_changes[zip_code].append({
                    'zip_code': zip_code,
                    'sequence': sequence,
                    'action': 'remove',
                    'pocket': pocket_num,
                    'insert': insert
                })
        
        # 4. Add new inserts to available pockets
        inserts_to_add = target_inserts - inserts_to_keep
        available_pockets = [p for p in range(1, 17) if p not in new_pocket_state]
        
        for idx, insert in enumerate(inserts_to_add):
            if idx < len(available_pockets):
                pocket_num = available_pockets[idx]
                pocket_changes[zip_code].append({
                    'zip_code': zip_code,
                    'sequence': sequence,
                    'action': 'add',
                    'pocket': pocket_num,
                    'insert': insert
                })
                new_pocket_state[pocket_num] = insert
        
        # Update pocket state for next iteration
        current_pockets = new_pocket_state.copy()
    
    return render_template(
        'operator_report.html',
        machine_id=machine_id,
        machine_name=machine_data['name'],
        assignments=machine_data['zips'],
        total_inserts=total_machine_inserts,
        pocket_changes=pocket_changes,
        report_date=datetime.now().strftime("%Y-%m-%d %H:%M")
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5001, debug=True)
