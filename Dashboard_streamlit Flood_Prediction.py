import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
from google.colab import drive
import streamlit as st

st.title("PREDICTION OF FLOODS")
df = pd.read_csv(r'/content/drive/MyDrive/Flood Prediction/flood_updated_1.csv')

# Group the data by MonsoonIntensity and calculate the average FloodProbability
flood_prob_by_intensity = df.groupby('MonsoonIntensity')['FloodProbability'].mean().reset_index()

# Visualization using Plotly
fig = px.bar(flood_prob_by_intensity, x='MonsoonIntensity', y='FloodProbability', 
             title='Average Flood Probability by Monsoon Intensity')
st.plotly_chart(fig)

# Group the data by 'TopographyDrainage' and calculate the average flood probability for each group
grouped_data = df.groupby('TopographyDrainage')['FloodProbability'].mean().reset_index()

# Create a bar chart to visualize the relationship
fig = px.bar(grouped_data, x='TopographyDrainage', y='FloodProbability', 
             title='Average Flood Probability by Topography Drainage')
st.plotly_chart(fig)

from sklearn.preprocessing import LabelEncoder

# Select columns to encode (excluding 'FloodProbability')
columns_to_encode = df.columns[df.columns != 'FloodProbability']

# Apply label encoding to each selected column
label_encoders = {}
for column in columns_to_encode:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  label_encoders[column] = le  # Store the encoder
  # Correlation Matrix
st.subheader("Correlation Matrix")
corr_matrix = df.corr()
fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                 x=corr_matrix.columns,
                                 y=corr_matrix.columns,
                                 colorscale='Viridis'))
st.plotly_chart(fig)

# Decode the columns
for column, le in label_encoders.items():
  df[column] = le.inverse_transform(df[column])

# Dropdown for Political Factor (with unique key)
selected_factor1 = st.selectbox('Select Political Factor', df['PoliticalFactors'].unique(), key='political_factor1')

# Filter data and calculate average flood probability
filtered_df1 = df[df['PoliticalFactors'] == selected_factor1]
average_probability1 = filtered_df1['FloodProbability'].mean()

st.write(f"Average Flood Probability for {selected_factor1}: {average_probability1:.2f}")

# Pie chart for selected political factor (with unique key)
labels = ['Flood Probability', 'No Flood Probability']
values = [average_probability1, 1 - average_probability1]
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_layout(title=f'Flood Probability Distribution for {selected_factor1}')
st.plotly_chart(fig, key='pie_chart1')


le_urbanization = LabelEncoder()
df['Urbanization_en'] = le_urbanization.fit_transform(df['Urbanization'])

# Create mapping of encoded values to original values
mapping = dict(zip(le_urbanization.classes_, le_urbanization.transform(le_urbanization.classes_)))

# Display the mapping in a dropdown
selected_encoding = st.selectbox('Select Urbanization Encoding:', options=list(mapping.keys()))
st.write(f"Encoded Value: {mapping[selected_encoding]}")

# Group the data by 'Urbanization' and calculate the average flood probability for each group
grouped_data = df.groupby('Urbanization_en')['FloodProbability'].mean().reset_index()

# Create a scatter plot to visualize the relationship
fig = px.scatter(grouped_data, x='Urbanization_en', y='FloodProbability', 
                 title='Average Flood Probability vs. Urbanization',
                 trendline="ols")  # Add a trendline
st.plotly_chart(fig)





label_encoders = {}
for column in columns_to_encode:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  label_encoders[column] = le 

# Value Prediction Model
st.header("Flood Prediction Model")

features = ["MonsoonIntensity", "TopographyDrainage", "RiverManagement", "DrainageSystems", "Deforestation", "Urbanization", "InadequatePlanning", "ClimateChange", "CoastalVulnerability", "Landslides"]
X = df[features]
y = df["FloodProbability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write(f"Model R-squared: {r2_score(y_test, y_pred):.4f}")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.coef_
})
feature_importance = feature_importance.sort_values("Importance", ascending=False)

fig_importance = px.bar(feature_importance, x="Importance", y="Feature", orientation="h")
fig_importance.update_layout(title="Feature Importance for Flood Prediction")
st.plotly_chart(fig_importance)# Value Prediction Model
#st.header("Flood Prediction Model")

features = ["MonsoonIntensity", "TopographyDrainage", "RiverManagement", "DrainageSystems", "Deforestation", "Urbanization", "InadequatePlanning", "ClimateChange", "CoastalVulnerability", "Landslides"]
X = df[features]
y = df["FloodProbability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)




columns = ['Urbanization', 'InadequatePlanning', 'MonsoonIntensity',
           'TopographyDrainage', 'RiverManagement', 'DrainageSystems',
           'Deforestation', 'ClimateChange', 'CoastalVulnerability', 'Landslides',
           'FloodProbability']

# Create a Streamlit app


# Scatter plots for Flood Probability vs. Contributing Factors
st.title("Relationship Between Flood Probability and Contributing Factors")
for col in columns[1:]:
    fig, ax = plt.subplots()
    sns.scatterplot(x='FloodProbability', y=col, data=df, ax=ax)
    plt.savefig(f"flood_probability_vs_{col}.png")
    st.pyplot(fig)





# Store label encoders for each column
label_encoders = {}

# Select columns to encode (excluding 'FloodProbability')
columns_to_encode = df.columns[df.columns != 'FloodProbability']

# Apply label encoding to each selected column
for column in columns_to_encode:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  label_encoders[column] = le  # Store the encoder




# Correlation heatmap
st.subheader("Correlation Between Variables")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, ax=ax)
plt.savefig("correlation_heatmap.png")
st.pyplot(fig)

