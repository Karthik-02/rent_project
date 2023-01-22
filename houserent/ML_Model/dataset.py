import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv("House_Rent_Dataset.csv")
print(data.head())

print(data.isnull().sum())
figure = px.bar(data, x=data["City"], y = data["Rent"], color = data["BHK"],title="Rent in Different Cities According to BHK")

with open('p_graph.html', 'a') as f:
    f.write(figure.to_html(full_html=False, include_plotlyjs='cdn'))




