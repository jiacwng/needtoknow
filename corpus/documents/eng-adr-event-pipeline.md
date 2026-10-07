+++
id = "eng-adr-event-pipeline"
title = "Architecture decision: event pipeline broker"
readers = ["group:engineering"]
fact = "NATS JetStream"
question = "Which message broker did engineering choose for the new event pipeline?"
allowed_user = "amir"
denied_user = "marco"
+++
Status: accepted, October 2026. Deciders: Amir Haddad, Rafael Ortega.

Context: customers want stock changes in real time. We need a broker that keeps messages for
a few days, so a customer can catch up after downtime.

Decision: we will use NATS JetStream. It is small to operate, keeps messages on disk and
supports one stream per customer. We compared it with Kafka and RabbitMQ. Kafka needed more
operational work than our team can give it. RabbitMQ did not fit the replay requirement as
well.

Consequences: Oskar Nilsson will run a three-node cluster in each region.
