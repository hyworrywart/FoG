# Freebase service

This guide is adapted from the [ToG Freebase guide](https://github.com/GasolSun36/ToG/blob/main/Freebase/README.md), with credit to the ToG authors and contributors. Consult that upstream guide for its download locations and detailed Virtuoso instructions.

The experiments use the final official Freebase RDF dump released before Freebase was discontinued, served with Virtuoso 7.2.5. The dump is not included in this repository.

1. Install Virtuoso 7.2.5 and ensure enough disk space for the database and bulk-load working files.
2. Obtain the final official Freebase RDF dump.
3. Configure Virtuoso bulk loading for the dump, load it, and checkpoint the database.
4. Start Virtuoso and verify that its SPARQL endpoint can query `http://rdf.freebase.com/ns/` identifiers.
5. Set `FREEBASE_ENDPOINT` if the service is not available at `http://localhost:8890/sparql`.

Example smoke query:

```sparql
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?name WHERE {
  ns:m.0d6lp ns:type.object.name ?name .
  FILTER(langMatches(lang(?name), "en"))
} LIMIT 1
```

The cache builder performs unrestricted two-hop neighborhood queries and can be expensive. Build caches once, outside benchmark timing, and retain the resulting `manifest.json`. Freebase content remains subject to its own source terms.
