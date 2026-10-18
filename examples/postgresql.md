# PostgreSQL

PostgreSQL is a relational database. Indexes speed up reads at the cost of slower writes.
Use EXPLAIN ANALYZE to see how a query is executed and whether an index is used.

VACUUM reclaims storage from dead tuples, and autovacuum normally handles it. Connection
pooling keeps the number of open connections small when many application workers run.
