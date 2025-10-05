# src/ingestion_spark.py
"""
Spark ingestion template.
- Batch read: reads CSV logs into DataFrame
- Streaming template: reads from Kafka topic `siem-logs` and writes to parquet or memory sink

Note: requires pyspark installed and Spark configured.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
import os

def batch_ingest(csv_path="../data/sample_logs.csv"):
    spark = SparkSession.builder.appName("SIEMIngestBatch").getOrCreate()
    df = spark.read.option("header", True).csv(csv_path)
    df.printSchema()
    df.show(5, truncate=False)
    # Write to parquet for downstream processing
    out = "../data/parquet_logs"
    df.write.mode("overwrite").parquet(out)
    print(f"[INFO] Wrote parquet to {out}")
    spark.stop()

def streaming_template(kafka_bootstrap="localhost:9092", topic="siem-logs"):
    spark = SparkSession.builder \
        .appName("SIEMStreaming") \
        .getOrCreate()
    raw = (spark.readStream
           .format("kafka")
           .option("kafka.bootstrap.servers", kafka_bootstrap)
           .option("subscribe", topic)
           .option("startingOffsets", "earliest")
           .load())
    # Assume value is JSON string with keys timestamp,host,event_type
    # schema = ...
    # df = raw.selectExpr("CAST(value AS STRING)")
    # parsed = df.select(from_json(col("value"), schema).alias("data")).select("data.*")
    # parsed.writeStream.format("parquet").option("path","../data/stream_parquet").option("checkpointLocation","../data/checkpoint").start()
    print("[INFO] Streaming template created. Fill schema and start.")
    # spark.streams.awaitAnyTermination()
