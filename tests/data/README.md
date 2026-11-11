# Small input examples

`features.fa`, `features.gtf` and `features.sam` describe five short transcripts:

- `plus`: a single-exon coding transcript with repeated tags. Its CDS encodes MKPGF.
- `minus`: the same transcript sequence on the negative strand, split over two
  exons. Its ATG crosses the exon boundary. Genomic TIS/LTS/TTS are 74/50/49.
- `noncoding`: a lowercase, N-containing lncRNA without an annotated CDS, but with
  a short potential ORF encoding MPK.
- `nonstop`: an annotated CDS without a stop codon on the transcript.
- `silent`: a transcript with neither alignment records nor positive predictions.

The SAM contains duplicate footprints, different footprint lengths, and reads
outside the accepted 20–40 nt interval. Expected accepted read counts are
4/2/1/1/0 respectively. It deliberately omits `silent` from the alignment header.
Tests convert this same SAM to BAM and check that both formats, and combined or
separate HDF5 storage, preserve those counts.

The larger GRCh38 snippet and existing saved models remain available for CLI
and prediction checks. Runtime outputs and indexes belong in pytest's temporary
directories, not alongside the checked-in input files.

`model_scores.json` records representative scores from the first fold of each
published human, mouse and RiboTIE model, evaluated with TRISTAN v1.1.1,
Python 3.11.9, PyTorch 2.7.1 and Lightning 2.4.0 on CPU. Sequence input is the
first 128 bases of the first transcript in `tt_predicting.h5`; RiboTIE input is
a footprint every third position in the 28-nt channel. Positions include
regularly spaced background samples and the eight highest scores. The model
test allows small floating-point differences (relative 1e-4, absolute 1e-7),
not changes to the model or its saved weights.
