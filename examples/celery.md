# Celery

Celery is a distributed task queue for Python. Workers consume jobs from a broker such as Redis
or RabbitMQ. Failed tasks can be retried with exponential backoff, and periodic jobs are
scheduled by the beat process.

Idempotent tasks are safe to retry. Pass identifiers instead of large objects so the message
stays small and the worker always reads fresh data.
