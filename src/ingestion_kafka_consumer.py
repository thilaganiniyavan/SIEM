# src/ingestion_kafka_consumer.py
"""
Optional: simple Kafka consumer that writes messages to CSV for batch processing.
Requires confluent_kafka or kafka-python.
"""

from kafka import KafkaConsumer
import json, csv, os

def consume_to_csv(bootstrap="localhost:9092", topic="siem-logs", out="../data/sample_logs_from_kafka.csv"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    consumer = KafkaConsumer(topic, bootstrap_servers=[bootstrap], auto_offset_reset='earliest', group_id="siem-group")
    with open(out, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","host","event_type"])
        for msg in consumer:
            try:
                payload = json.loads(msg.value.decode())
                writer.writerow([payload.get("timestamp"), payload.get("host"), payload.get("event_type")])
            except Exception as e:
                print("Parse error", e)
