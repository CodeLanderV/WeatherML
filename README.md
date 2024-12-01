# WeatherML
Basic Weather Prediction Model

- This is A basic Weather Prediction Model which uses SVM to determine if it will rain or not.
- I have considered in parameters like Season, temperature, Humdiity, Cloudy or not.

# Methodology:
- Load and prepare the dataset using pandas.
- Visualize the data with seaborn to get an understanding of patterns.
- Preprocess the data: Use LabelEncoder for categorical variables.
- Scale numerical data(like Temperature, Humidity, WindSpeed) using MinMaxScaler.
- Split the dataset into training and testing using train_test_split, 80% to 20% ratio
- Train the model using SVM.
- Evaluate the model using classification_report to assess its performance.

 # Performance Report:
 
               precision    recall  f1-score   support

           0       1.00      0.33      0.50         3
           1       0.00      0.00      0.00         0

    accuracy                           0.33         3
   macro avg       0.50      0.17      0.25         3
weighted avg       1.00      0.33      0.50         3
