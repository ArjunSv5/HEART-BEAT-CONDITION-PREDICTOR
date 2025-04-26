#Libraries req.
import pandas as pd  
import numpy as np
import os  
import librosa 
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#csv file imported to pandas
csv_file = 'C:/Users/Arjun S V/Desktop/HEART BEAT DATASET/set_a.csv'
df = pd.read_csv(csv_file) 

base_path = 'C:/Users/Arjun S V/Desktop/HEART BEAT DATASET/set_a'  
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)  
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  
        return np.mean(mfccs, axis=1)  
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None  

#features is x and labels is y when it comes to table format
features = []
labels = []

for index, row in df.iterrows():
    file_name = row['fname'].strip()  
    file_path = os.path.join(base_path, os.path.basename(file_name)) 
    label = row['label']  
    if pd.notna(label):  # Skip unlabeled data
        feature = extract_features(file_path)  
        if feature is not None:  
            features.append(feature)  
            labels.append(label) 


#features to X and labels to Y (in numpy array format)
X = np.array(features)  
y = np.array(labels)    


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\tARTIFACT\tEXTRAHLS\tMURMURS\t\tNORMAL")
print("ARTIFACT : ",conf_matrix[0][0],"\t\t",conf_matrix[0][1],"\t\t",conf_matrix[0][2],"\t\t",conf_matrix[0][3])
print("EXTRAHLS : ",conf_matrix[1][0],"\t\t",conf_matrix[1][1],"\t\t",conf_matrix[1][2],"\t\t",conf_matrix[1][3])
print("MURMURS : ",conf_matrix[2][0],"\t\t",conf_matrix[2][1],"\t\t",conf_matrix[2][2],"\t\t",conf_matrix[2][3])
print("NORMAL : ",conf_matrix[3][0],"\t\t",conf_matrix[3][1],"\t\t",conf_matrix[3][2],"\t\t",conf_matrix[3][3])
print("\n📋 Classification Report:\n", class_report)

