# Sleep Disorder Predictions

This project aims to predict sleep disorders based on lifestyle and health factors using machine learning techniques.

**DISCLAIMER:** The predictions made by this model are not proper medical diagnoses. Please use the app/model at your own discretion. See a doctor for more details.

## Project Structure

- `01-sleep-lifestyle-predictors-eda.ipynb`: Exploratory Data Analysis (EDA) of the dataset.
- `02-sleep-lifestyle-predictors-feature-select.ipynb`: Feature selection process to identify key predictors.
- `03-sleep-lifestyle-predictors-modeling.ipynb`: Model training and evaluation.
- `04-sleep-lifestyle-predictors-deploy.ipynb`: Deployment strategies for the predictive model.
- `sleep_predict_app.py`: Flask web application for user interaction and predictions.
- `sleep_health_lifestyle.csv`: Original dataset containing sleep health and lifestyle information.
- `encoded_sleep_health.csv`: Preprocessed dataset with encoded categorical variables.
- `key_features_sleep_health.csv`: Dataset containing selected key features.
- `scaler.joblib`: Scaler object used for data normalization.
- `sleep_disorder_model.joblib`: Trained machine learning model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sophiaachungg/sleep-disorder-predictions.git
2. Navigate to the project directory:
   ```bash
   cd sleep-disorder-predictions
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
4. Install the required packages:
   ```bash
   pip install -r requirements.txt

# Usage
Data Preprocessing: Run the notebooks in the following order to preprocess data and train the model:

* `01-sleep-lifestyle-predictors-eda.ipynb`
* `02-sleep-lifestyle-predictors-feature-select.ipynb`
* `03-sleep-lifestyle-predictors-modeling.ipynb`
Model Deployment: Use the `sleep_predict_app.py` script to launch the Flask web application:
```bash
python sleep_predict_app.py
```
Access the application at http://127.0.0.1:5000/ in your web browser.

Alternatively, you can run the application on Streamlit, as demonstrated in `04-sleep-lifestyle-predictors-deploy.ipynb`.

Dependencies
The project requires the following Python packages:

* Flask
* Jupyter
* scikit-learn
* pandas
* numpy
* joblib
For a complete list, refer to the requirements.txt file.

**Notes:**

- Ensure that the `requirements.txt` file is placed in the root directory of the project.
- The versions specified in `requirements.txt` are indicative. It's advisable to use the versions that are compatible with your development environment.
- Before deploying the Flask application, ensure that all necessary model and scaler files (`sleep_disorder_model.joblib` and `scaler.joblib`) are present in the project directory.

By following the above documentation and requirements, users should be able to set up and run the sleep disorder prediction application effectively.

