import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add after US_REGIONS definition
REGION_COORDINATES = {
    'West': {'lat': 40.5, 'lon': -120.5},
    'South': {'lat': 32.7, 'lon': -83.5},
    'Northeast': {'lat': 42.5, 'lon': -72.5},
    'Midwest': {'lat': 41.5, 'lon': -93.5},
    'Non-US': {'lat': 35.0, 'lon': 0.0}  # Default coordinates for Non-US
}

STATE_COORDINATES = {
    'AL': {'lat': 32.806671, 'lon': -86.791130},
    'AK': {'lat': 61.370716, 'lon': -152.404419},
    'AZ': {'lat': 33.729759, 'lon': -111.431221},
    'AR': {'lat': 34.969704, 'lon': -92.373123},
    'CA': {'lat': 36.116203, 'lon': -119.681564},
    'CO': {'lat': 39.059811, 'lon': -105.311104},
    'CT': {'lat': 41.597782, 'lon': -72.755371},
    'DE': {'lat': 39.318523, 'lon': -75.507141},
    'FL': {'lat': 27.766279, 'lon': -81.686783},
    'GA': {'lat': 33.040619, 'lon': -83.643074}
}

def generate_sample_data(seed=42):
    np.random.seed(seed)
    
    # Define locations
    us_states = ['NY', 'CA', 'IL', 'MA', 'WA', 'TX', 'FL']
    countries = ['United States', 'United Kingdom', 'Germany', 'France', 'Japan', 'Australia']
    
    # Define region mapping
    US_REGIONS = {
        'Northeast': ['NY', 'MA'],
        'West': ['CA', 'WA'],
        'Midwest': ['IL'],
        'South': ['TX', 'FL']
    }
    
    # Create reverse mapping from state to region
    STATE_TO_REGION = {state: region for region, states in US_REGIONS.items() for state in states}
    
    # Function to get region
    def get_region(country, state):
        if country == 'United States':
            return STATE_TO_REGION.get(state, 'Other US')
        return 'Non-US'
    
    # Generate dates for 2024
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    
    # Generate base employee data
    n_employees = 1000
    employees = []
    
    for employee_id in range(n_employees):
        # Assign initial location
        is_us = np.random.choice([True, False], p=[0.6, 0.4])
        initial_location = {
            'country': 'United States' if is_us else np.random.choice([c for c in countries if c != 'United States']),
            'state': np.random.choice(us_states) if is_us else None
        }
        
        # Determine if employee will move during the year (20% chance)
        will_move = np.random.random() < 0.2
        if will_move:
            # Randomly choose when the move will happen (between month 3 and 9)
            move_month = np.random.randint(3, 10)
            
            # Determine new location
            is_moving_international = np.random.random() < 0.3
            if is_us:
                if is_moving_international:
                    new_location = {
                        'country': np.random.choice([c for c in countries if c != 'United States']),
                        'state': None
                    }
                else:
                    new_location = {
                        'country': 'United States',
                        'state': np.random.choice([s for s in us_states if s != initial_location['state']])
                    }
            else:
                if is_moving_international:
                    new_location = {
                        'country': np.random.choice([c for c in countries if c != initial_location['country'] and c != 'United States']),
                        'state': None
                    }
                else:
                    new_location = {
                        'country': 'United States',
                        'state': np.random.choice(us_states)
                    }
        
        # Generate employee base data
        base_engagement = np.random.normal(0.75, 0.15)
        base_salary = np.random.normal(100000, 30000)
        tenure_years = np.random.exponential(3)
        tenure_range = 'High' if tenure_years > 5 else 'Medium' if tenure_years > 2 else 'Low'
        reporting_manager = f"Manager {np.random.randint(1, 6)}"
        
        # Generate monthly data
        for i, date in enumerate(dates):
            current_location = new_location if (will_move and i >= move_month) else initial_location
            
            # Add some random variation to metrics
            engagement = min(1, max(0, base_engagement + np.random.normal(0, 0.05)))
            salary = base_salary * (1 + np.random.normal(0, 0.02))
            
            # Add coordinates based on region
            region = get_region(current_location['country'], current_location['state'])
            if current_location['country'] == 'United States':
                if current_location['state'] in STATE_COORDINATES:
                    lat = STATE_COORDINATES[current_location['state']]['lat']
                    lon = STATE_COORDINATES[current_location['state']]['lon']
                else:
                    # Use region coordinates if state coordinates not available
                    lat = REGION_COORDINATES[region]['lat']
                    lon = REGION_COORDINATES[region]['lon']
            else:
                # Use Non-US coordinates for international employees
                lat = REGION_COORDINATES['Non-US']['lat']
                lon = REGION_COORDINATES['Non-US']['lon']
            
            employees.append({
                'date': date,
                'employee_id': employee_id,
                'engagement': engagement,
                'salary': salary,
                'tenure_range': tenure_range,
                'reporting_manager': reporting_manager,
                'country': current_location['country'],
                'state': current_location['state'],
                'region': region,
                'latitude': lat,
                'longitude': lon
            })
    
    df = pd.DataFrame(employees)
    return df

if __name__ == "__main__":
    df = generate_sample_data()
    df.to_csv('employee_data.csv', index=False)
    print("Sample data generated and saved to 'employee_data.csv'") 