# Stock Price Analysis and Prediction

This project involves analyzing historical stock prices and building predictive models to forecast future prices using Python. The project includes data collection, exploration, visualization, preparation, modeling, and evaluation.

## Project Structure

- **notebooks/**: Contains Jupyter notebooks for data analysis and model building.
  - `Stock_Price_Analysis.ipynb`: Main notebook for stock price analysis and model building.

- **scripts/**: Contains Python scripts for specific tasks.
  - `real_time_prediction.py`: Script for making real-time stock price predictions using the trained models.

- **models/**: Directory to store trained machine learning models.
  - `scaler.pkl`: The scaler used for normalizing the data.
  - `stock_price_predictor_lr.pkl`: The trained Linear Regression model for stock price prediction.

- **LICENSE**: License file for the project.
- **README.md**: Overview of the project, setup instructions, and documentation.
- **requirements.txt**: List of Python packages required to run the project.

## Getting Started

### Prerequisites

Make sure you have Python and the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- yfinance
- xgboost
- jupyter
- joblib

### Installation

1. Clone the repository

```bash
git clone https://github.com/trnabab/Stock_Price_Analysis_and_Prediction.git
cd Stock_Price_Analysis_and_Prediction

```
2. Set up a virtual environment

``` bash
python -m venv venv
source venv\Scripts\activate  # On Mac use `venv\bin\activate`
pip install -r requirements.txt
```
3. Launch Jupyter Notebook

```bash
jupyter notebook
```
4. Open the Notebook:
Open the `Stock_Price_Analysis.ipynb` notebook and follow the steps.

## Project Steps

### 1. Setup Environment
Install and import the necessary libraries for the project.

### 2. Data Collection
Download historical stock price data for Microsoft (MSFT) from Yahoo Finance using the `yfinance` library.

### 3. Data Exploration and Visualization
Explore the data to understand its structure and plot the closing price over time. Calculate and plot moving averages to identify trends.

### 4. Data Preparation
Create new features such as returns and volatility. Handle missing values and split the data into training and testing sets. Normalize the data for modeling.

### 5. Model Building and Training
We will build and train multiple machine learning models to predict stock prices:

- **Linear Regression Model**: Used as the initial model to predict the next day's closing price.
- **Random Forest Classifier**: An ensemble learning method for regression.
- **Gradient Boosting**: Builds an additive model in a forward stage-wise manner.
- **XGBoost**: An optimized distributed gradient boosting library.

### 6. Model Evaluation
Evaluate the performance of each model using several metrics to understand their performance:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

### Real-time Prediction
The `scripts/real_time_prediction.py` script uses the trained models to make real-time stock price predictions.

### Scripts
The `scripts` directory contains the following script:
- `real_time_prediction.py`: Script for making real-time stock price predictions using the trained models.

### Models
The `models` directory contains the following models:
- `scaler.pkl`: The scaler used for normalizing the data.
- `stock_price_predictor_lr.pkl`: The trained Linear Regression model for stock price prediction.

## Conclusion
Summarized the findings and discuss the model's performance. Suggested potential improvements and future work.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This project was inspired by the need to analyze and predict stock prices using machine learning techniques.
Special thanks to the developers of the libraries used in this project.