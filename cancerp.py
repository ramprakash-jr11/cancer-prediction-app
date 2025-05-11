import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (assuming it's saved as a CSV file or directly from the provided text)
# For this example, I'll assume the dataset is saved as 'cancer_data.csv'
# If you have it as a string or different format, let me know, and I can adjust accordingly.
data = pd.read_csv(r"C:\Users\Sivaram S\Desktop\mp3\archive\The_Cancer_data_1500_V2.csv")

# Display the first few rows to verify the data
print("Dataset Preview:")
print(data.head())

# Features and target
X = data.drop('Diagnosis', axis=1)  # All columns except 'Diagnosis'
y = data['Diagnosis']               # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to get user input and predict
def predict_cancer():
    print("\nEnter the following parameters for cancer prediction:")
    
    # Collect user input for each feature
    age = float(input("Age (e.g., 58): "))
    gender = int(input("Gender (0 = Female, 1 = Male): "))
    bmi = float(input("BMI (e.g., 16.08): "))
    smoking = int(input("Smoking (0 = No, 1 = Yes): "))
    genetic_risk = int(input("Genetic Risk (0 = Low, 1 = Medium, 2 = High): "))
    physical_activity = float(input("Physical Activity (hours/week, e.g., 8.14): "))
    alcohol_intake = float(input("Alcohol Intake (units/week, e.g., 4.14): "))
    cancer_history = int(input("Cancer History (0 = No, 1 = Yes): "))
    
    # Create a numpy array with the user input
    user_input = np.array([[age, gender, bmi, smoking, genetic_risk, 
                            physical_activity, alcohol_intake, cancer_history]])
    
    # Make prediction
    prediction = rf_model.predict(user_input)[0]
    
    # Output the result
    if prediction == 1:
        print("\nPrediction: Cancer (1)")
    else:
        print("\nPrediction: No Cancer (0)")

# Run the prediction function
while True:
    predict_cancer()
    again = input("\nWould you like to predict again? (yes/no): ").lower()
    if again != 'yes':
        break

print("Thank you for using the cancer prediction tool!")
