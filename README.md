
# Heart Condition Predictor

LIVE WEBSITE LINK : https://heart-beat-condition-predictor-uuuvkakyccsr3ura7kgm5d.streamlit.app/

This project builds a machine learning model to predict heart conditions based on heartbeat audio recordings using feature extraction and classification techniques.

## Project Overview
- **Dataset:** Heartbeat audio recordings (`set_a`) and their labels from a CSV file.
- **Feature Extraction:** MFCC (Mel Frequency Cepstral Coefficients) features extracted using Librosa.
- **Model Used:** Random Forest Classifier.
- **Performance:** Prints model accuracy, confusion matrix, and a full classification report.

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `os`
  - `librosa`
  - `scikit-learn`

Install the required libraries using:
```bash
pip install pandas numpy librosa scikit-learn
```

## How It Works
1. **Load Dataset:** Read the CSV containing filenames and labels.
2. **Extract Features:** For each heartbeat audio file, extract MFCC features and compute their mean.
3. **Prepare Data:** Organize extracted features (`X`) and labels (`y`) into numpy arrays.
4. **Train-Test Split:** Split the data into training and testing sets.
5. **Model Training:** Train a Random Forest Classifier on the training data.
6. **Evaluation:** Evaluate model performance using accuracy score, confusion matrix, and classification report.

## How to Run
1. Make sure the dataset path (`set_a` folder and `set_a.csv`) are correctly set in the code.
2. Run the Python file:
```bash
python main_program.py
```
3. View the model's accuracy, confusion matrix, and detailed classification metrics.

## Output Example
- ✅ Model Accuracy: 92.00%
- 📋 Confusion Matrix and classification report printed to the console.

## Notes
- Ensure that all audio files mentioned in the CSV are available in the specified `base_path`.
- You can tune the model parameters or try different classifiers for further improvement.
