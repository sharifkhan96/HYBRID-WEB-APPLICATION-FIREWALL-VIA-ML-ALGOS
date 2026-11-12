# IEEE Access submission package

`paper_ieee_access.tex` — the ML benchmark paper, reformatted from the
conference `IEEEtran` version into IEEE Access's actual required class
(`ieeeaccess.cls`, official template downloaded from
`ieeeaccess.ieee.org`, not a community copy). Compiles to a real 7-page PDF
with the correct IEEE Access layout: logo, colored ABSTRACT/INDEX TERMS
banners, `\history`/DOI placeholders, and author biography sections.

**One local-only patch**: `ieeeaccess.cls` has been edited in one spot
(search for "PATCHED" — the Pantone spot-color block for the "accessblue"
banner color) because it depends on raw pdfTeX object primitives
(`\pdfobj`, `\pdfliteral`) that the local compiler (`tectonic`) doesn't
implement; replaced with a plain CMYK color definition. This only affects
what the color looks like in *your own local preview* — IEEE's production
system typesets the accepted manuscript from your submitted source using
their own real pipeline, so this patch has no effect on the submitted or
published version. If you compile on Overleaf (which runs real pdfTeX),
you can use the **original, unpatched** `ieeeaccess.cls` from the official
zip instead and the Pantone color will render correctly there too.

The pipeline diagram (Fig. 2) is a pre-rendered PNG
(`figures/pipeline_diagram.png`) instead of inline TikZ, because `tikz`/
`pgf` has a separate, content-independent incompatibility with this class
under `tectonic` (reproduced with a two-node test picture — not specific to
this figure). If you compile on Overleaf, inline TikZ should work fine
there if you'd rather have it live; the source for it is preserved in
`../ieee/paper_ieee.tex`'s Fig. 2.

## Before you submit — three things only you can fill in

1. **Abid Naseri's affiliation** — search for `AFFILIATION NEEDED HERE` (one
   occurrence, in the `\address[2]{...}` line).
2. **Both author biographies** — search for `Author biography to be
   completed`. IEEE Access requires a short (100–200 word) biography for
   every author, placed after the references. I can't write these
   honestly since I don't know your degrees, prior roles, or research
   interests beyond what's in this project — please fill them in yourself
   or tell me the facts and I'll draft the prose.
3. **An ORCID iD** for the submitting/corresponding author (Sharif), linked
   to and visible on your ScholarOne account — IEEE Access requires this at
   submission. Register free at [orcid.org](https://orcid.org) if you
   don't have one yet.

## Recompiling after edits

```bash
tectonic paper_ieee_access.tex
```

(or upload this whole folder to Overleaf — it's self-contained: class,
fonts, style files, and figures are all included).
