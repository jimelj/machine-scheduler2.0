# User Documentation - Production Scheduler

## Introduction

The Production Scheduler is a web application designed to optimize production scheduling across 3 machines, each with 16 pockets that can hold inserts. The system schedules zip codes to machines in a way that maximizes insert continuity between consecutive zip codes, improving operational efficiency by minimizing the need to change inserts between jobs.

## Getting Started

### Accessing the Application

The Production Scheduler can be accessed through your web browser at the URL provided by your system administrator.

### Navigation

The application has a simple navigation menu at the top of the page with the following options:

- **Home**: Main dashboard and overview
- **Upload Files**: Upload your weekly Insert Orders file
- **Generate Schedule**: Create a new production schedule
- **View Schedule**: View the current production schedule

## Workflow

### Step 1: Upload Insert Orders File

1. Click on "Upload Files" in the navigation menu
2. Click "Choose File" and select your weekly Insert Orders CSV file
3. Click "Upload" to process the file
4. Wait for confirmation that the file has been uploaded successfully

**Note**: The system will automatically filter for "CBA LONG ISLAND" entries in the Insert Orders file.

### Step 2: Generate Schedule

1. Click on "Generate Schedule" in the navigation menu
2. Select which day(s) to include in the schedule:
   - All Days: Schedule both Monday and Tuesday
   - Monday Only: Schedule only Monday zip codes
   - Tuesday Only: Schedule only Tuesday zip codes
3. Click "Generate Schedule" to create an optimized production schedule
4. Wait for confirmation that the schedule has been generated successfully

### Step 3: View Schedule

1. Click on "View Schedule" in the navigation menu
2. The schedule overview shows:
   - Total ZIP codes scheduled
   - Total inserts reused
   - Machine assignments
   - Reuse efficiency chart
3. Click "View Machine Details" for any machine to see detailed information

### Step 4: Machine Details

The machine details page provides:

1. **ZIP Code Sequence**: Shows the sequence of zip codes assigned to the machine
   - Use "Previous ZIP" and "Next ZIP" buttons to navigate through the sequence
   - Each zip code shows how many inserts are reused from the previous zip code

2. **Machine Pockets Visualization**: Shows a visual representation of the 16 pockets
   - Green pockets: Inserts that need to be added
   - Red pockets: Inserts that need to be removed
   - Blue pockets: Inserts that can be kept from the previous zip code
   - Gray pockets: Empty pockets

3. **Pocket Change Instructions**: Provides step-by-step instructions for operators
   - Lists which inserts to remove from which pockets
   - Lists which inserts to add to which pockets

## Understanding the Schedule

### Insert Continuity

The scheduling algorithm maximizes "insert continuity," which means it tries to reuse as many inserts as possible between consecutive zip codes on each machine. This reduces the number of insert changes needed, improving operational efficiency.

### Machine Assignments

Zip codes are assigned to machines based on which machine will allow for the most insert reuse. The first zip code for each machine is assigned randomly, but subsequent zip codes are assigned to maximize continuity.

### Pocket Assignments

Each machine has 16 pockets that can hold different inserts. The system tracks which inserts are in which pockets and provides instructions for changes needed between zip codes.

## Tips for Operators

1. **Follow the Sequence**: Process zip codes in the order shown in the ZIP Code Sequence
2. **Check Pocket Visualization**: Use the pocket visualization to see which inserts need to be changed
3. **Follow Change Instructions**: Use the step-by-step instructions to make the necessary changes
4. **Verify Completion**: After making changes, verify that the correct inserts are in the correct pockets

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

### Getting Help

If you encounter issues that you cannot resolve, please contact your system administrator or IT support team for assistance.

## Glossary

- **Insert/Copy**: Raw advertisement received to be put together into one package
- **Pocket**: A slot in the machine that can hold one insert type
- **ZIP Code**: The first 5 digits of the ZipRoute field
- **Insert Continuity**: The reuse of inserts between consecutive zip codes
- **Machine Assignment**: The assignment of a zip code to a specific machine
- **Pocket Assignment**: The assignment of an insert to a specific pocket
