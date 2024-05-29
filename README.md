# Stock Price Analysis and Prediction

This project involves analyzing historical stock prices and building a predictive model to forecast future prices using Python. The project includes data collection, exploration, visualization, preparation, modeling, and evaluation.

## Project Structure

- **1. Setup Environment**: Installing and importing necessary libraries.
- **2. Data Collection**: Downloading historical stock price data using `yfinance`.
- **3. Data Exploration and Visualization**: Exploring and visualizing data to identify trends and patterns.
- **4. Data Preparation**: Creating new features, handling missing values, and splitting the data.
- **5. Model Building and Training**: Building and training a linear regression model.
- **6. Model Evaluation**: Evaluating model performance using MSE, MAE, and R² metrics.

## Getting Started

### Prerequisites

Make sure you have Python and the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- yfinance

### Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/Stock_Price_Analysis_and_Prediction.git
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
Download historical stock price data for Tesla (TSLA) from Yahoo Finance using the `yfinance` library

### 3. Data Exploration and Visualization
Explore the data to understand its structure and plot the closing price over time. Calculate and plot moving averages to identify trends.

### 4. Data Preparation
Create new features such as returns and volatility. Handle missing values and split the data into training and testing sets. Normalize the data for modeling.

### 5. Model Building and Training
Build and train a linear regression model to predict stock prices. Fit the model using the training data.

### 6. Model Evaluation
Evaluate the model's performance using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) metrics. Interpret the results to assess the model's accuracy and reliability.

## Conclusion
Summarize the findings and discuss the model's performance. Suggest potential improvements and future work.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- This project was inspired by the need to analyze and predict stock prices using machine learning techniques
- Special thanks to the developers of the libraries used in this project.