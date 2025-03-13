import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Example Dataset
data = pd.DataFrame({
    "Feature1": [1, 2, 3, 4, 5, 6],
    "Feature2": [1, 2, 3, 4, 5, 6],
    "Class": [0, 0, 1, 1, 0, 1]
})

# Train a Random Forest model
X = data[["Feature1", "Feature2"]]
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Define the function to be used by Gradio
def predict(feature1, feature2):
    try:
        # Debugging: Print values of feature1 and feature2
        print(f"Received features: Feature 1 = {feature1}, Feature 2 = {feature2}")
        
        # Make prediction
        prediction = model.predict([[feature1, feature2]])
        
        # Debugging: Print the prediction result
        print(f"Prediction: {prediction[0]}")
        
        return f"Predicted Class: {prediction[0]}"
    
    except Exception as e:
        # Handle errors and return a message
        print(f"Error: {e}")
        return f"Error: {str(e)}"

# Create the Gradio Interface
iface = gr.Interface(fn=predict, 
                     inputs=[gr.Slider(minimum=0, maximum=10, label="Feature 1"), 
                             gr.Slider(minimum=0, maximum=10, label="Feature 2")],
                     outputs="text")

iface.launch(share=True)  # Optionally, set share=True to generate a public link
