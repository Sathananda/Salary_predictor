# Salary Predictor

## Overview
A machine learning project that predicts employee salaries based on their years of experience. This project demonstrates fundamental concepts in regression analysis and data science using Python.

## Description
The Prediction of Salaries of Employees based on their past experience (in years).

## Features
- **Data Analysis**: Explore and understand salary trends across experience levels
- **Machine Learning Model**: Implement regression algorithms to predict salaries
- **Visualization**: Plot data and model predictions for better insights
- **Easy to Use**: Simple interface for training and making predictions

## Tech Stack
- **Language**: Python 3.x
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning algorithms
  - `matplotlib` / `seaborn` - Data visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sathananda/Salary_predictor.git
cd Salary_predictor
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. **Prepare Your Data**: Ensure your dataset contains experience (in years) and corresponding salary information

2. **Train the Model**:
```python
python train_model.py
```

3. **Make Predictions**:
```python
python predict.py --experience 5
```

## Dataset Format

The dataset should be a CSV file with the following structure:

| Experience (Years) | Salary |
|---|---|
| 1 | 30000 |
| 2 | 35000 |
| 3 | 40000 |
| ... | ... |

## Project Structure

```
Salary_predictor/
├── README.md
├── requirements.txt
├── data/
│   └── salary_data.csv
├── models/
│   └── trained_model.pkl
├── train_model.py
├── predict.py
└── visualize.py
```

## How It Works

1. **Data Loading**: Load salary data from CSV
2. **Feature Engineering**: Prepare experience as input feature and salary as target variable
3. **Model Training**: Use linear regression or other algorithms to fit the model
4. **Prediction**: Use trained model to predict salaries for given experience levels
5. **Visualization**: Plot actual vs predicted values

## Example

```python
from salary_predictor import SalaryPredictor

# Initialize and train model
predictor = SalaryPredictor()
predictor.train('data/salary_data.csv')

# Make a prediction
predicted_salary = predictor.predict(years_experience=5)
print(f"Predicted salary for 5 years experience: ${predicted_salary:.2f}")
```

## Model Performance

- Evaluate model accuracy using metrics like:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R-squared (R²) Score

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

## License

This project is open source and available under the MIT License.

## Author

**Sathananda**

## Contact & Support

For questions or issues, please open an issue on the [GitHub repository](https://github.com/Sathananda/Salary_predictor).

## Roadmap

- [ ] Add multiple regression models (Ridge, Lasso, etc.)
- [ ] Implement cross-validation
- [ ] Add web interface
- [ ] Support for non-linear relationships
- [ ] Deploy as API

---

**Last Updated**: June 2026
