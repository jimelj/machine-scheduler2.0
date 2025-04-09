"""
Installation and Deployment Guide for Production Scheduler

This document provides instructions for installing and deploying the Production Scheduler application.
"""

# Installation Guide

## Prerequisites

Before installing the Production Scheduler, ensure you have the following:

- Python 3.10 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)
- Access to the command line/terminal

## Local Installation

### Step 1: Get the Code

Either clone the repository using Git:

```bash
git clone https://github.com/yourusername/production-scheduler.git
cd production-scheduler
```

Or download and extract the ZIP file from the provided source.

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Initialize the Database

```bash
python -c "from app import app, db; app.app_context().push(); db.create_all()"
```

### Step 5: Prepare Data Files

1. Place the "Zips by Address File Group.xlsx" file in the `upload` directory.
2. Create the uploads directory if it doesn't exist:

```bash
mkdir -p uploads
```

### Step 6: Run the Application

```bash
python app.py
```

The application will be available at http://localhost:5000

## Production Deployment

For production deployment, we recommend using Gunicorn as the WSGI server and Nginx as the reverse proxy.

### Step 1: Install Gunicorn

```bash
pip install gunicorn
```

### Step 2: Create a Gunicorn Configuration File

Create a file named `gunicorn_config.py`:

```python
bind = "0.0.0.0:8000"
workers = 3
timeout = 120
```

### Step 3: Run with Gunicorn

```bash
gunicorn -c gunicorn_config.py app:app
```

### Step 4: Configure Nginx (Optional)

Install Nginx and create a configuration file:

```
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Step 5: Set Up Systemd Service (Optional)

Create a systemd service file for automatic startup:

```
[Unit]
Description=Production Scheduler Gunicorn Daemon
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/path/to/production-scheduler
ExecStart=/path/to/production-scheduler/venv/bin/gunicorn -c gunicorn_config.py app:app
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable production-scheduler
sudo systemctl start production-scheduler
```

## Docker Deployment (Alternative)

### Step 1: Create a Dockerfile

Create a file named `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads

EXPOSE 5000

CMD ["python", "app.py"]
```

### Step 2: Build and Run the Docker Image

```bash
docker build -t production-scheduler .
docker run -p 5000:5000 -v $(pwd)/upload:/app/upload production-scheduler
```

## Updating the Application

To update the application:

1. Pull the latest code or extract the new ZIP file
2. Install any new dependencies: `pip install -r requirements.txt`
3. Restart the application

## Backup and Restore

### Database Backup

The application uses SQLite by default, which stores the database in a file. To backup:

```bash
cp production_scheduler.db production_scheduler.db.backup
```

### Database Restore

To restore from a backup:

```bash
cp production_scheduler.db.backup production_scheduler.db
```

## Troubleshooting

### Common Issues

1. **Application won't start**:
   - Check that all dependencies are installed
   - Verify Python version is 3.10 or higher
   - Check for error messages in the console

2. **Database errors**:
   - Ensure the database file is writable
   - Try reinitializing the database

3. **File upload issues**:
   - Check that the uploads directory exists and is writable
   - Verify the file format is correct (CSV for Insert Orders)

### Logs

Check the application logs for more detailed error information:

- When running directly: Check the console output
- When running as a systemd service: `sudo journalctl -u production-scheduler`
- When running with Docker: `docker logs <container_id>`
