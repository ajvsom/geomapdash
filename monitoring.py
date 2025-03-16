import time
import requests
import logging
from datetime import datetime, timezone
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

class AppMonitor:
    def __init__(self, app_url):
        self.app_url = app_url
        self.start_time = datetime.now(timezone.utc)
        self.uptime_log_file = "logs/uptime.log"
        self.status_log_file = "logs/status.log"
        
        # Initialize log files
        for log_file in [self.uptime_log_file, self.status_log_file]:
            if not os.path.exists(log_file):
                with open(log_file, 'w') as f:
                    f.write("timestamp,status\n")
    
    def check_status(self):
        """Check if the app is responsive"""
        try:
            start_time = time.time()
            response = requests.get(self.app_url)
            response_time = time.time() - start_time
            
            status = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': response.status_code,
                'response_time': response_time
            }
            
            # Log status
            with open(self.status_log_file, 'a') as f:
                f.write(f"{status['timestamp']},{status['status']},{status['response_time']:.2f}\n")
            
            return status
            
        except requests.RequestException as e:
            logger.error(f"Error checking status: {e}")
            return None
    
    def log_uptime(self):
        """Log current uptime"""
        current_time = datetime.now(timezone.utc)
        uptime = current_time - self.start_time
        
        with open(self.uptime_log_file, 'a') as f:
            f.write(f"{current_time.isoformat()},{uptime.total_seconds()}\n")
    
    def get_instance_hours(self):
        """Calculate total instance hours"""
        if not os.path.exists(self.uptime_log_file):
            return 0
        
        total_hours = 0
        with open(self.uptime_log_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                timestamp, uptime = line.strip().split(',')
                total_hours += float(uptime) / 3600  # Convert seconds to hours
        
        return total_hours
    
    def get_status_summary(self):
        """Get a summary of app status"""
        if not os.path.exists(self.status_log_file):
            return {}
        
        total_checks = 0
        successful_checks = 0
        total_response_time = 0
        
        with open(self.status_log_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                total_checks += 1
                _, status, response_time = line.strip().split(',')
                if int(status) == 200:
                    successful_checks += 1
                total_response_time += float(response_time)
        
        return {
            'total_checks': total_checks,
            'availability': (successful_checks / total_checks * 100) if total_checks > 0 else 0,
            'avg_response_time': total_response_time / total_checks if total_checks > 0 else 0
        }

# Example usage
if __name__ == '__main__':
    monitor = AppMonitor('http://localhost:8050')
    
    # Check current status
    status = monitor.check_status()
    if status:
        print(f"App Status: {status}")
    
    # Log uptime
    monitor.log_uptime()
    
    # Get instance hours
    hours = monitor.get_instance_hours()
    print(f"Total instance hours: {hours:.2f}")
    
    # Get status summary
    summary = monitor.get_status_summary()
    print(f"Status Summary: {summary}") 