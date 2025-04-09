# Machine Scheduler

A Flask-based web application for optimizing production scheduling across multiple machines with 16 pockets each, maximizing insert continuity between consecutive zip codes.

## Features

- Upload weekly Insert Orders file
- Filter CBA LONG ISLAND entries automatically
- Match ZIP codes with Address File Groups for MON/TUE scheduling
- Optimize machine assignments to maximize insert continuity
- Visual interface for viewing schedules and pocket assignments
- Real-time pocket change calculations and instructions

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd machineScheduler3
```

2. Create and activate a virtual environment:
```bash
python -m venv venv_py310
source venv_py310/bin/activate  # On Windows: venv_py310\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5001
```

3. Upload your Insert Orders file and follow the on-screen instructions to generate and view schedules.

## Development

The application is built with:
- Python 3.10
- Flask
- Pandas for data processing
- Bootstrap for UI
- JavaScript for dynamic interactions

## Project Structure

- `app.py` - Main Flask application
- `scheduling_algorithm.py` - Core scheduling logic
- `templates/` - HTML templates
- `static/` - CSS, JavaScript, and static assets
- `uploads/` - Directory for temporary file storage (not tracked in git)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Weekly Workflow

1. Upload new order file each week
2. Generate schedule with appropriate parameters
3. View and print schedule and machine details
4. The system will replace the previous week's data with the new schedule

## License

This project is proprietary software. All rights reserved.

## Contact

For questions or support, please contact the developer.

---

Developed by [Jimel J. Joseph](https://github.com/jimelj)
