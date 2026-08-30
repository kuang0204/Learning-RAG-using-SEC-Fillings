A RAG system over 10 SEC 10-K filings (Apple, Amazon, Dell, Alphabet, Meta, Microsoft, NVIDIA FY24/25/26, Palantir).

Full write-up: RAG_REPORT_LEARNINGS.html

This project is used to teach myself RAG so it is a walkthrough implementing enhancement by enhancement to see each effect live. 
The goal is to see the theory in practice, rather than building the whole pipeline at once.

Key findings

A two-company comparison query collapsed into a single blended query embedding, so one company's language dominated dense retrieval and the other got zero chunks in the top 10. Fixed with query decomposition and round-robin merging of sub-query results.
Embedding systematically underranked tables relative to prose. Traced to how tables were serialized into text, a documented failure mode known as the "linearization bottleneck." Fixed with header-prepended row serialization, pairing each value inline with its column label.
BM25 hurt retrieval. Adding BM25 unexpectedly degraded results. Hypothesis: corpus homogeneity — 10 filings on the same regulatory template mean supposedly rare terms like "total revenue" and "fiscal year" appear in nearly every document, collapsing BM25's IDF.

Results:

Same corpus, embedding model, and gold set throughout — only the pipeline changes. recall@20 0.60 → 0.95, all-gold@10 0.05 → 0.45 against the naive baseline.

Refusal: 19/20 unanswerable questions correctly refused, zero hallucinated figures, across 3 runs.
