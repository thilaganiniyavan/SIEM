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

Folder structure
text
hybrid_siem_full/
│
├── data/
│   ├── sample_logs.csv            # synthetic logs (generated)
│   ├── features.csv               # extracted features
│   └── dbn_model.pt               # saved trained model
│
├── src/
│   ├── generate_logs.py           # synthetic log generator
│   ├── ingestion_spark.py         # Spark batch + streaming ingestion templates
│   ├── ingestion_kafka_consumer.py# Kafka consumer template (optional)
│   ├── features.py                # feature engineering (incl. graph features)
│   ├── models/                   # RBM, DBN, Autoencoder implementations
│   ├── trainer.py                # orchestrates pretraining, fine-tuning, evaluation
│   ├── evaluator.py              # evaluation metrics and ROC plotting
│   ├── llm_rule_optimizer.py     # LLM-based rule optimization module
│   ├── feedback_loop.py          # collects analyst feedback and triggers retrain
│   ├── api/
│   │   ├── app.py                # FastAPI service: scoring and feedback endpoints
│   │   └── schemas.py            # Pydantic schemas for API request validation
│   └── utils.py                 # helper functions (logging, graph features)
│
├── docker/
│   ├── Dockerfile                # Docker image for API service
│
├── requirements.txt             # Python dependencies
└── README.md                    


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