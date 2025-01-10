# Credit Scoring and Customer Segmentation
This project focuses on developing a customer credit scoring model and segmenting customers based on their purchasing behavior. The aim is to analyze customer data to predict loyalty and improve decision-making by using data-driven insights.

# Key Features:
Data Preprocessing: Cleaned and prepared a dataset of 19,148 customer records, addressing missing values, invalid data, and outliers.
Customer Segmentation: Utilized Recency, Frequency, and Monetary (RFM) analysis to segment customers into loyalty groups.
Clustering: Applied K-Means clustering to group customers based on their behavior and characteristics.
Prediction Model: Built a K-Nearest Neighbors (KNN) classifier to predict customer loyalty with an accuracy of 79%.
Real-Time Application: Integrated the predictive model into a Flask-based web application, allowing real-time predictions on customer loyalty.
# How It Works:
Data Cleaning: The dataset was cleaned by removing duplicates, handling missing values, and filtering outliers to ensure the data quality.
RFM Analysis: Recency, Frequency, and Monetary values were used to calculate customer behavior, helping to identify patterns in spending and activity.
Clustering: K-Means clustering was used to segment customers into different groups based on their RFM values.
KNN Classification: A KNN model was trained to predict customer loyalty levels based on their historical data, helping businesses target high-value customers.
Web Integration: The final model was deployed using Flask, providing a user-friendly interface for real-time predictions.
# Insights:
Retention Strategy: By identifying high-value customer segments, businesses can focus retention efforts on their most loyal and profitable customers.
Marketing Focus: Clustering helps businesses tailor their marketing campaigns to specific customer groups, improving relevance and engagement.
Loyalty Prediction: The KNN model forecasts customer loyalty, enabling proactive measures to reduce churn.
Optimized Resources: Insights into customer behavior ensure that resources are allocated effectively, maximizing returns on marketing and customer service efforts.

# Technologies Used:
Python
Pandas
Scikit-learn
Flask
Matplotlib
Seaborn
Numpy

# Conclusion:
This project demonstrates how customer segmentation and predictive modeling can be applied to analyze customer behavior and drive business decisions. By using data to understand customer loyalty, businesses can optimize marketing strategies, enhance customer retention, and make more informed decisions. The actionable insights from the project include identifying high-value customer segments for targeted retention strategies, such as personalized offers or loyalty rewards, to maintain engagement with profitable customers. The customer segments formed through K-Means clustering enable more effective marketing campaigns, focusing on high-potential or loyal groups. The KNN model forecasts customer loyalty, helping businesses anticipate churn and take preventive actions for at-risk customers. Additionally, insights into customer loyalty allow for better allocation of marketing and service resources, ultimately improving ROI.
