# Importing necessary libraries
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from dash import Dash, html, dcc, dash_table
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
from itertools import combinations
from collections import Counter
import io
import base64
import matplotlib.pyplot as plt

# Function to prepare data
def prepare_data(file_path):
    df = pd.read_csv(file_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Preprocess data
    duplicate_rows = df[df.duplicated()]
    df['medication_name'] = df['medication_name'].astype(str).str.lower().str.split(',')
    df['symptoms'] = df['symptoms'].fillna("").astype(str).str.strip()
    df = df[df['symptoms'] != ""]
    df['symptom_count'] = df['symptoms'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
    df['symptoms'] = df['symptoms'].str.replace(r'\s*\([^)]*\)', '', regex=True).str.split().str[:3].str.join(" ")
    
    return df, duplicate_rows

# Function to generate Word Cloud
def generate_wordcloud(symptom_counts):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(symptom_counts)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return img

# Load and prepare data
file_path = 'dataset\\merge_demo_amos_v3.csv'
df, duplicate_rows = prepare_data(file_path)

# Data Preparation for Visualizations
source_counts = df['source'].value_counts().reset_index()
source_counts.columns = ['Source', 'Number of Occurrences']

disease_counts = df['disease_name'].value_counts().head(10).reset_index()
disease_counts.columns = ['Disease', 'Number of Occurrences']

medication_counts = df['medication_name'].value_counts().head(10).reset_index()
medication_counts.columns = ['Medication', 'Number of Occurrences']

medication_list = [med.strip() for sublist in df['medication_name'].dropna() for med in sublist]
individual_med_counts = pd.Series(medication_list).value_counts()
individual_med_counts = individual_med_counts[individual_med_counts.index != 'nan'].head(10).reset_index()
individual_med_counts.columns = ['Medication', 'Number of Occurrences']

symptom_counts = pd.Series([s.strip() for sublist in df['symptoms'].str.lower().str.split(',')
                            for s in sublist if s.strip()]).value_counts()
symptom_counts.index = symptom_counts.index.str.replace('\n', ' ')
wordcloud_img = generate_wordcloud(symptom_counts)

top_symptom_counts = df[['disease_name', 'symptom_count']].sort_values(by='symptom_count', ascending=False).head(10)

co_occurrence = Counter()
for symptoms in df['symptoms'].dropna().str.lower().str.split(','):
    for combo in combinations(sorted(set(symptoms)), 2):
        co_occurrence[combo] += 1
co_df = pd.DataFrame(co_occurrence.items(), columns=['Symptom Pair', 'Count']).sort_values(by='Count', ascending=False).head(10)
co_df['Symptom Pair'] = co_df['Symptom Pair'].apply(lambda x: f"{x[0]} - {x[1]}")

vectorizer = CountVectorizer(stop_words='english', max_features=50)
bow_matrix = vectorizer.fit_transform(df['description'].dropna())
common_words = pd.DataFrame({'word': vectorizer.get_feature_names_out(),
                             'count': bow_matrix.toarray().sum(axis=0)}).sort_values(by='count', ascending=False).head(10)

# Network Preparation
edges = [(row['disease_name'], symptom.strip())
         for _, row in df.iterrows()
         for symptom in row['symptoms'].lower().split(',') if symptom.strip()]
G = nx.Graph()
G.add_edges_from(edges)
filtered_nodes = [node for node, degree in G.degree() if degree > 5]
G_filtered = G.subgraph(filtered_nodes)
elements = [{'data': {'id': node, 'label': node}, 'classes': 'disease' if node in df['disease_name'].values else 'symptom'}
            for node in G_filtered.nodes()]
elements.extend({'data': {'source': edge[0], 'target': edge[1]}} for edge in G_filtered.edges())

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Cytoscape Stylesheet
stylesheet = [
    {'selector': 'node', 'style': {'content': 'data(label)', 'font-size': '12px', 'text-valign': 'center', 'text-halign': 'center'}},
    {'selector': '.disease', 'style': {'background-color': '#FF6347', 'shape': 'ellipse', 'width': 60, 'height': 60}},
    {'selector': '.symptom', 'style': {'background-color': '#4682B4', 'shape': 'rectangle', 'width': 50, 'height': 30}},
    {'selector': 'edge', 'style': {'curve-style': 'bezier', 'line-color': '#CCC', 'width': 2}}
]

# Capture df.info() output
info_buffer = io.StringIO()
df.info(buf=info_buffer)
info_str = info_buffer.getvalue()

# Dash Layout
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Medical Data Analysis Dashboard", className="text-center mb-4"))),

    # Data Overview Section
    dbc.Row([
        dbc.Col([
            html.H3("Dataset Info"),
            html.Pre(info_str, style={'font-size': '12px'}),
            html.H4("Duplicate Rows"),
            dash_table.DataTable(
                data=duplicate_rows.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in duplicate_rows.columns],
                style_table={'overflowX': 'auto'},
                page_size=5
            )
        ], width=6),
        dbc.Col([
            html.H3("Data Sources"),
            dcc.Graph(figure=px.bar(source_counts, x='Number of Occurrences', y='Source',
                                    title="Distribution of Data Sources",
                                    color='Number of Occurrences', color_continuous_scale='Blues'))
        ], width=6)
    ], className="mb-4"),

    # Disease and Medication Section
    dbc.Row([
        dbc.Col([
            html.H3("Top 10 Diseases"),
            dcc.Graph(figure=px.bar(disease_counts, x='Number of Occurrences', y='Disease',
                                    title="Top 10 Diseases by Occurrence",
                                    color='Number of Occurrences', color_continuous_scale='Reds')),
            dash_table.DataTable(
                data=disease_counts.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in disease_counts.columns],
                style_table={'overflowX': 'auto'}
            )
        ], width=6),
        dbc.Col([
            html.H3("Top 10 Medications"),
            dcc.Graph(figure=px.bar(medication_counts, x='Number of Occurrences', y='Medication',
                                    title="Top 10 Medications by Occurrence",
                                    color='Number of Occurrences', color_continuous_scale='Greens')),
            dash_table.DataTable(
                data=medication_counts.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in medication_counts.columns],
                style_table={'overflowX': 'auto'}
            )
        ], width=6)
    ], className="mb-4"),

    # Individual Medications and Symptoms Section
    dbc.Row([
        dbc.Col([
            html.H3("Top 10 Individual Medications"),
            dcc.Graph(figure=px.bar(individual_med_counts, x='Number of Occurrences', y='Medication',
                                    title="Top 10 Most Frequent Individual Medications",
                                    color='Number of Occurrences', color_continuous_scale='Blues')),
            dash_table.DataTable(
                data=individual_med_counts.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in individual_med_counts.columns],
                style_table={'overflowX': 'auto'}
            )
        ], width=6),
        dbc.Col([
            html.H3("Symptoms Word Cloud"),
            html.Img(src=f"data:image/png;base64,{wordcloud_img}", style={'width': '100%'})
        ], width=6)
    ], className="mb-4"),

    # Symptom Analysis Section
    dbc.Row([
        dbc.Col([
            html.H3("Top 10 Diseases by Symptom Count"),
            dcc.Graph(figure=px.bar(top_symptom_counts, x='symptom_count', y='disease_name',
                                    title="Top 10 Diseases by Symptom Count",
                                    color='symptom_count', color_continuous_scale='Purples')),
            dash_table.DataTable(
                data=top_symptom_counts.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in top_symptom_counts.columns],
                style_table={'overflowX': 'auto'}
            )
        ], width=6),
        dbc.Col([
            html.H3("Top 10 Symptom Co-occurrences"),
            dcc.Graph(figure=px.bar(co_df, x='Count', y='Symptom Pair',
                                    title="Top 10 Symptom Co-occurrences",
                                    color='Count', color_continuous_scale='Oranges')),
            dash_table.DataTable(
                data=co_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in co_df.columns],
                style_table={'overflowX': 'auto'}
            )
        ], width=6)
    ], className="mb-4"),

    # Description and Network Section
    dbc.Row([
        dbc.Col([
            html.H3("Top 10 Common Words in Descriptions"),
            dcc.Graph(figure=px.bar(common_words, x='count', y='word',
                                    title="Top 10 Common Words in Descriptions",
                                    color='count', color_continuous_scale='Viridis')),
            dash_table.DataTable(
                data=common_words.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in common_words.columns],
                style_table={'overflowX': 'auto'}
            )
        ], width=6),
        dbc.Col([
            html.H3("Disease-Symptom Network (>5 Connections)"),
            cyto.Cytoscape(
                id='cytoscape-network',
                elements=elements,
                layout={'name': 'cose', 'idealEdgeLength': 100, 'nodeOverlap': 20},
                style={'width': '100%', 'height': '600px'},
                stylesheet=stylesheet
            )
        ], width=6)
    ])
], fluid=True)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
