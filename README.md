Hybrid SIEM Full Prototype
Overview
This project implements a hybrid Security Information and Event Management (SIEM) pipeline that leverages scalable data ingestion, advanced feature engineering, deep learning models, and large language model (LLM)-based rule optimization to improve real-time threat detection and reduce alert fatigue.

It includes:

Scalable batch and streaming data ingestion templates (Spark, Kafka)

Feature engineering including summary, binary, temporal, and relational graph-based features

Unsupervised pretraining using Deep Belief Networks (DBN) or autoencoders, followed by supervised fine-tuning

Integrated model evaluation and monitoring

LLM-driven SIEM rule optimization engines supporting OpenAI API and HuggingFace models

Analyst feedback integration with automated retraining trigger

FastAPI microservice exposing scoring and rule suggestion endpoints

Docker containerization for easy deployment




Quickstart: Running Locally
Set up environment:

bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
Generate synthetic logs:

bash
python src\generate_logs.py
Extract features:

bash
python src\features.py
Train the model (autoencoder by default):

bash
python src\trainer.py
To enable DBN pretraining, edit trainer.py accordingly.

Launch API service:

bash
uvicorn src.api.app:app --reload
Use API:

Visit http://127.0.0.1:8000/docs for interactive Swagger UI.

Test /score endpoint to get risk predictions.

Submit analyst feedback on /feedback.

Features
Scalable Ingestion: Batch and streaming ingestion from CSV, Kafka, or Spark.

Sophisticated Feature Engineering: Includes event counts, binary flags, temporal stats, and PageRank-based relational features.

Hybrid Modeling: Unsupervised pretraining with DBN or Autoencoder, followed by supervised fine-tuning for risk classification.

LLM Rule Optimization: AI-assisted suggestions to merge, drop, or refine detection rules and reduce alert overload.

Feedback Loop: Analyst input stored and used to trigger retraining, enabling continuous model improvement.

Microservice API: Easy integration into existing workflows with FastAPI and Docker support.

Extending to Production
Replace synthetic logs with real-time SIEM exports or Kafka streams.

Use persistent scaler storage and model versioning (e.g., MLflow).

Implement sandboxed rule testing before deploying LLM-generated rule changes.

Deploy on cluster infrastructure for scalability (e.g., Kubernetes, EMR).

Troubleshooting
Ensure Python environment activated and dependencies installed.

Relative imports require __init__.py files in packages (src, src/api, src/models).

Adjust file paths relative to your working directory.

Monitor API and training logs for errors.

License
MIT License
