# Caching

A cache stores computed results so repeated requests are cheap. The hard part is invalidation:
entries must expire or be replaced when the underlying data changes.

A version-keyed cache avoids manual invalidation. Include a data version in every key, and
bumping the version makes older entries unreachable.
