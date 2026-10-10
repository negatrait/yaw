It would be highly beneficial to run tests that assert the real behavior of source code. Tautological tests are a poison that dilute the test credibility from two angles:
1. Creating a tautological test gives a false signal of functioning source code.
2. Tautological tests bloat the codebase and the context for coding agents, inducing context rot.

Allways test the real source code, never do soft mocking.
