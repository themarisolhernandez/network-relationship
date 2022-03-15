# Objective

Every day, a massive volume of text data is created. Every 24 hours, 500 million Tweets are published, 18.7 billion text messages are sent, and 4.4 million new blogs are written. As the number of data points increases, so do the relationships that potentially exist between these data points. No tweet, text message, or academic paper exists in a vacuum; they are all interconnected, whether via the people who follow them, the messages they receive, the topics they discuss, or the sources they cite.

Extracting such data in its entirety can be difficult and time-consuming, not to mention expensive. Whether it be a researcher who is reviewing prior related publications or a team of attorneys assessing every text, email, or phone conversation, an investigation like these might take months, or even years, to complete, depending on the complexity of the case and the resources required.

There is little use in spending so much time and money on such an endeavor. Fortunately, text analysis can be achieved at a much faster rate, and an improved accuracy, with the use of contemporary computer technologies such as natural language processing (NLP). These technologies, however, cannot fully replace the human domain. Thus, NLP tools and its outputs are most effective when used in conjunction with domain experts.

When it comes to such deployments, visualizations are a critical component. In order to better explore, manage, and comprehend a dataset as well as the results of NLP analysis, an effective visualization is necessary. In this investigation, I use [Dash Cytoscape](https://github.com/plotly/dash-cytoscape), a tool that allows you to explore and visualize datasets and the connections between them. Using [Plotly Dash](https://plotly.com/dash/), I develop an app that demonstrates how Dash Cytoscape can be used to visualize hundreds of academic publications that have been categorized into topics that were developed by [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) techniques. These papers are also connected by citations, truly illustrating a network.

# About the Data
