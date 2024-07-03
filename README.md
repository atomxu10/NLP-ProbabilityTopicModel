# Probability Topic Model: Model Comparison for Different Text Sizes ![Language](https://img.shields.io/badge/language-Python-green)

This project aimed to introduce, explain, and use probabilistic topic models, a class of statistical models for text data analysis designed to identify and extract latent topic structures from text data. This article focused on the **LDA (Latent Dirichlet Allocation) model** in the probabilistic topic model, which is based on Bayesian statistical inference methods and uses the Dirichlet distribution to model the relationship between topics and documents. The main objective of the LDA model is to extract topics from textual data to enhance our comprehension of the content and to facilitate tasks like text classification.

The LDA model in the project was tested on three news dataset sizes: **title text, body text, and full content** to investigate the impact of text data size. The evaluation criteria for model quality were **classification efficiency** and **coherence score**.

## Content
- [1. Introduction & Overview](#1)
- [2. Latent Dirichlet Allocation (LDA) Theory](#2)
  - [2.1 Bayesian Inference](#2-1)
  - [2.2 The Generative Process of LDA Model](#2-2)
  - [2.3 Parameters Fitting](#2-3)
  - [2.4 Assumption, Notation, and Algorithm](#2-4)
  - [2.5 Gibbs Sampling](#2-5)
- [3. Method](#3)
  - [3.1 Data](#3-1)
  - [3.2 Model Construction](#3-2)
    - [3.2.1 Pre-processing of Data](#3-2-1)
    - [3.2.2 Model Specification](#3-2-2)
    - [3.2.3 Label Topics](#3-2-3)
  - [3.3 Classification Analysis](#3-3)
    - [3.3.1 Topics Visualisation](#3-3-1)
    - [3.3.2 Prediction Accuracy](#3-3-2) 

## 1. Introduction & Overview

With the development of the internet, information is provided to people in various forms, such as texts, images, audio, and videos. However, as the information continues to grow, the necessity of developing an efficient tool to process, comprehend, and analyze this information becomes essential. Consequently, in the field of text information retrieval, extensive research is conducted in areas including text summarisation, text extraction, and dimensionality reduction. Researchers believe that if a substantial amount of textual data is converted into some low-dimensional subspaces through some tools, information covering this type of text data can be harvested, and then it can be "zoomed in or out" to access and explore the information we desire. Among them, the probabilistic topic model is one of the models with rapid development and mature application.

The probabilistic topic model is an unsupervised analytical tool to uncover hidden thematic structures from text and to establish probabilistic relationships between texts and topics. The LDA topic model is one of the most common and widely applied probabilistic topic models. It is characterized by its Bayesian foundation, which aids in revealing the distribution of topics in textual data, the relationships between topics, and the connections between topics and vocabulary. Practical applications of probabilistic topic models included text classification, information retrieval, content analysis, and content modeling, among other fields. By unearthing the underlying themes within textual data, the model could offer a deeper comprehension and analysis of textual content.

This project attempted to investigate the use of the LDA model and evaluate its quality. Particularly, it intends to answer three questions:

**Q1: Does the size of the text have an impact on the recognition and classification of the model when using the LDA topic model as a classifier?**

**Q2: How can hyper-parameters be utilized to improve model efficiency?**

**Q3: How does the size of the text data affect the interpretability of the LDA model?**











