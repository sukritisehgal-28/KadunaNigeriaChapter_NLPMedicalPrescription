 The script leverages Pandas for data manipulation, Plotly for interactive charts, 
 Dash for web-based dashboards, and other tools for advanced visualizations like word clouds and network graphs.


The Installation Methods
There are two primary methods to install these packages, catering to different user preferences for setup:
Individual Installation via pip:
Users can install each package with specific version constraints using pip commands.
For example:
pip install pandas plotly>=4.0.0 wordcloud scikit-learn networkx dash>=2.0.0 dash-cytoscape dash-bootstrap-components matplotlib
This method is straightforward but can be tedious for multiple packages.
Using a requirements.txt File:
For better reproducibility, especially in team settings or deployment, create a requirements.txt file with the following content:



To set up the environment:
Ensure Python is installed (preferably 3.8 or higher for compatibility with recent package versions).
Create a new directory for your project and place the script and requirements.txt (if using) in it.
Open a terminal or command prompt, navigate to the directory, and run either the individual pip commands or pip install -r requirements.txt.
Verify installation by running pip list to see installed packages and their versions, ensuring plotly>=4.0.0 and dash>=2.0.0 are present.
Run the script, and access the dashboard at http://127.0.0.1:8050 in your browser.


