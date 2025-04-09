# Production Scheduler Documentation

## Overview

The Production Scheduler is a web application designed to optimize production scheduling across 3 machines, each with 16 pockets that can hold inserts. The system schedules zip codes to machines in a way that maximizes insert continuity between consecutive zip codes, improving operational efficiency by minimizing the need to change inserts between jobs.

## Key Features

- **Balanced Machine Utilization**: Distributes work evenly across all three machines
- **Insert Continuity Optimization**: Maximizes the reuse of inserts between consecutive zip codes
- **Visual Pocket Representation**: Shows operators which inserts to add, remove, or keep
- **Detailed Change Instructions**: Provides step-by-step guidance for pocket changes
- **Day-Specific Scheduling**: Supports scheduling for Monday, Tuesday, or both days

## Scheduling Algorithm

The scheduling algorithm has been specifically designed to address two key requirements:

1. **Proper utilization of all three machines**: The algorithm distributes zip codes across all three machines using a combination of strategies:
   - Initial distribution of zip codes with the most inserts to each machine
   - Round-robin assignment with continuity optimization
   - Machine balance correction to ensure even distribution

2. **Maximizing insert continuity**: The algorithm prioritizes insert reuse by:
   - Finding the next zip code with maximum insert overlap for each machine
   - Tracking which inserts can be kept, which need to be added, and which need to be removed
   - Calculating and displaying reuse efficiency statistics

## User Guide

### Accessing the Application

The Production Scheduler can be accessed at:
[http://5000-i21057ymy1lygcvgczvif-8ca236b0.manus.computer](http://5000-i21057ymy1lygcvgczvif-8ca236b0.manus.computer)

### Workflow

1. **Upload Insert Orders File**
   - Navigate to "Upload Files" in the menu
   - Select your weekly Insert Orders CSV file
   - Click "Upload" to process the file

2. **Generate Schedule**
   - Navigate to "Generate Schedule" in the menu
   - Select which day(s) to include (Monday, Tuesday, or All Days)
   - Click "Generate Schedule" to create an optimized production schedule

3. **View Schedule**
   - Review the schedule summary showing total zip codes and inserts reused
   - Check the machine assignments and reuse efficiency for each machine
   - Click "View Machine Details" for any machine to see detailed information

4. **Machine Details**
   - Use the "Previous ZIP" and "Next ZIP" buttons to navigate through the sequence
   - View the pocket visualization showing which inserts to add, remove, or keep
   - Follow the pocket change instructions for each zip code

### Understanding the Visualization

The machine detail page uses color-coding to help operators understand what changes are needed:

- **Green pockets**: Inserts that need to be added
- **Red pockets**: Inserts that need to be removed
- **Blue pockets**: Inserts that can be kept from the previous zip code
- **Gray pockets**: Empty pockets

## Technical Details

### Improved Scheduling Algorithm

The scheduling algorithm has been enhanced to ensure proper utilization of all three machines:

1. **Initial Distribution**: The algorithm starts by distributing zip codes with the most inserts to each machine.

2. **Round-Robin Assignment**: It then uses a round-robin approach to assign remaining zip codes, while still prioritizing insert continuity.

3. **Machine Balance Correction**: After the initial assignment, the algorithm checks for imbalances and redistributes work if necessary:
   - Identifies machines with too few assignments (less than 70% of average)
   - Identifies machines with too many assignments (more than 130% of average)
   - Moves assignments from overloaded machines to underutilized ones

4. **Continuity Optimization**: Throughout the process, the algorithm maximizes insert continuity by:
   - Calculating overlap between consecutive zip codes
   - Finding the best next zip code for each machine
   - Tracking which inserts can be reused

### System Requirements

- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection
- No special software installation required

## Troubleshooting

### Common Issues

1. **No Schedule Generated**:
   - Ensure the Insert Orders file has been uploaded
   - Check that the file contains entries for "CBA LONG ISLAND"
   - Verify that the ZIP codes in the Insert Orders file match those in the Zips by Address File Group

2. **Missing Inserts**:
   - Check that the Insert Orders file has the required columns
   - Ensure that AdvertiserAccount, StoreName, and ZipRouteVersion are properly formatted

3. **Browser Issues**:
   - Try refreshing the page
   - Clear your browser cache
   - Try using a different browser

## Support

For additional assistance, please contact the system administrator.
