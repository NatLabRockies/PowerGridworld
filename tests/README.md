This directory uses PyTest.  From the repo root:

```
uv run pytest tests/
```

Tests are minimal for now, checking basic imports, env creation and
stepping using baseline min/max/random policies, plus a few targeted
regression tests.  There is no checking for "physics" or "correctness"!
