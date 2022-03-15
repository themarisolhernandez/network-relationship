# Objective
In this investigation, I use [Dash Cytoscape](https://github.com/plotly/dash-cytoscape), a tool that allows you to explore and visualize datasets and the connections between them. Using [Plotly Dash](https://plotly.com/dash/), I develop an app that demonstrates how Dash Cytoscape can be used to visualize hundreds of academic publications that have been categorized into topics developed by [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) techniques. These papers are also connected by citations, truly illustrating a network diagram.

# About the Data
For this demonstration, 693 papers from the University of the Pacific Thesis/Dissertation [database](https://scholarlycommons.pacific.edu/uop_etds/) were scraped and categorised into 10 topics using LDA analysis.

# Dash Cytoscape
Dash Cytoscape is built to help visualize and explore relationships.

In the app, each node (i.e. paper) is placed based on its topic composition as determined by the LDA analysis, with edges representing citation relationships and colored according to the primary topic group of the cited paper; node sizes calculated so that the more oft-cited papers are larger in size.

The app allows additional filtering with its interactive features. Users can filter by minimum citation(s) and department. Additionally, you can choose to show/hide the edges indicating citation connections.

Node (or indeed, edge) selections are also sources of further interactivity, in this case revealing paper details.

You may have also noticed a slider on the right side of the animation above. It is configured to modify the t-SNE parameter ‘perplexity’. In Dash, this is achieved by setting up a slider element and listening for a change to it in a callback function, before passing it on as a parameter to perform updated calculations.
