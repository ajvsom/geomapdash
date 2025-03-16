# Employee Analytics Dashboard

This is a Dash web application that visualizes employee data across different dimensions including engagement, salary, and headcount. The application provides interactive filters and visualizations to analyze employee metrics across different locations and reporting managers.

## Features

- **Interactive Filters**:
  - Manager selection
  - Metric selection (Engagement, Salary, Headcount)
  - Location selection

- **Visualizations**:
  - Geographic heatmap showing selected metrics by country
  - Line chart showing trends by tenure range
  - Bar chart showing US vs non-US employee distribution

## Data Structure

The application uses sample data with the following structure:
- Employee ID
- Engagement (binary: 0/1, updated quarterly)
- Salary (updated monthly)
- Location (city and country)
- Tenure Range (high, medium, low)
- Reporting Manager

## Setup and Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   pip install dash-bootstrap-components
   ```

3. Generate sample data:
   ```bash
   python data_generator.py
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to `http://127.0.0.1:8050/`

## Usage

1. Use the dropdown menus at the top of the dashboard to:
   - Select a specific manager or view all managers
   - Choose the metric to visualize (Engagement, Salary, or Headcount)
   - Filter by location

2. The dashboard will automatically update to show:
   - A geographic heatmap of the selected metric
   - A line chart showing the metric trends by tenure range
   - A bar chart showing the US vs non-US employee distribution

## Data Updates

- Salary, location, and reporting manager data is updated monthly
- Engagement data is updated quarterly
- The dataset covers the period from January 31, 2024 to December 31, 2024