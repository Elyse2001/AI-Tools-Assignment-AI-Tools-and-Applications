# AI-Tools-Assignment-AI-Tools-and-Applications
assignment 
Part 1: Theoretical Understanding (40%)
Q1. TensorFlow vs. PyTorch

Programming style

TensorFlow: Graph-based execution (eager mode added later). Good for large-scale production with TensorFlow Serving & TensorFlow Lite.

PyTorch: Dynamic computation graph (define-by-run). Feels more “pythonic,” easier for research, debugging, and experimentation.

Ecosystem

TensorFlow: Strong deployment ecosystem (TensorFlow Serving, TensorFlow.js, TF Lite, TFX pipelines).

PyTorch: Tight PyData integration; TorchServe + ONNX export for deployment.

Choice

Use PyTorch when prototyping, researching, or needing flexibility/debugging.

Use TensorFlow when deploying at scale, needing mobile/edge support, or production-ready pipelines.

Q2. Jupyter Notebook use cases in AI

Exploratory data analysis (EDA): Interactive plots, checking distributions, correlations, quick model testing.

Prototyping & Documentation: Mix code, results, and markdown explanations → great for research papers, teaching, or sharing ML workflows.

Q3. spaCy vs. basic Python string operations

Basic string ops: Can split, replace, lower, or search substrings. Limited to surface-level text manipulation.

spaCy: Provides linguistic annotations (POS tags, dependency parsing, lemmatization), pretrained NER models, and tokenization optimized for multiple languages. This enables real NLP tasks beyond string matching.

Comparative Analysis: Scikit-learn vs. TensorFlow
Aspect	Scikit-learn	TensorFlow
Target applications	Classical ML (SVM, decision trees, clustering, regression)	Deep learning (CNNs, RNNs, transformers)
Ease of use	Very beginner-friendly; consistent API	Steeper learning curve; more boilerplate
Community support	Mature, stable, widely used in academia & industry for small/medium ML tasks	Huge community for DL, strong support from Google, many tutorials for neural networks
streamlit run app.py

