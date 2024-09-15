# Titanic Survival Prediction

I used a Random Forest Classifier to predict survival on the Titanic dataset from kaggle competition. The goal is to predict whether a passenger survived or not based on Sex,Age,Passenger fare and others.It has 0.74162 accuracy.

## Files

- `train.csv`: The training dataset with features and survival labels.
- `test.csv`: The test dataset with features but without survival labels.
- `titanic_predictor.py`: The Python script that trains the model and generates predictions.
- `submission_random_forest.csv`: The output file with predictions for the test dataset.

## Requirements

To run the script, ensure you have the following Python libraries installed:

- `pandas`
- `scikit-learn`

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn
