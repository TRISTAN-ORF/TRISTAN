# Quick Start

## 👋 About

TRISTAN (TRanslational Identification Suite using Transformer Networks for ANalysis) is a set of tools aimed at detecting ORFs that are translated in organisms. It achieves this through analysis of sequence context and/or ribosome profiling data.

TRISTAN tools are built using the newest advances and best practices in machine learning, step away from manually curated rules for data processing (Let the optimization algorithm handle it!), and are designed with flexibility and modularity in mind.

To illustrate the advantages and design principles of TRISTAN, consider the following:

* **Unbiased Data Utilization**: TRISTAN does not rely on custom positive or negative training sets or synthetic data. Instead, it leverages the full transcriptome, allowing the models to learn complex translational patterns directly from biological data without pre-imposed biases.
* **Robust Model Validation**: To ensure the generalizability and robustness of our models, training, validation, and test sets are separated by chromosome to prevent information leakage and provide a more accurate assessment of performance on unseen data.
* **Data-Driven Decision Making**: Unlike traditional approaches that often employ hardcoded rules for data alteration or prediction adjustments (e.g., specific rules for multicistronic transcripts, start codon usage, etc.), TRISTAN's machine learning models learn these nuances intrinsically. This data-driven approach allows for more adaptive and accurate predictions across diverse biological contexts.
* **Comprehensive Performance Evaluation**: Model performance and benchmarking have been evaluated using established metrics such as Precision-Recall (PR) AUC and Receiver Operating Characteristic (ROC) AUC scores, providing a clear and quantitative understanding of TRISTAN's predictive power.
* **Seamless Downstream Integration**: TRISTAN generates various output file formats designed for easy integration with common downstream analysis tools. Furthermore, output file format customizability allows users to tailor the results to their specific research pipelines and preferences.

The package `transcript-transformer` incorporates the functionality of TIS Transformer, ([paper](https://doi.org/10.1093/nargab/lqad021), [repository](https://github.com/TRISTAN-ORF/TIS_transformer)) and RiboTIE ([paper](https://doi.org/10.1101/2023.06.20.545724), [repository paper](https://github.com/TRISTAN-ORF/RiboTIE_article), [repository tool](https://github.com/TRISTAN-ORF/RiboTIE)). TRISTAN makes use of the [Performer](https://arxiv.org/abs/2009.14794) architecture to allow for the annotations and processing of transcripts at single nucleotide resolution. The package applies `h5py` for data loading and `pytorch-lightning` as a high-level interface for training and evaluation of deep learning models. `tristan-orf` is designed to allow a high degree of modularity.


## 🛠️ Installation

`PyTorch` is used as the deep learning library. Follow the instructions [here](https://pytorch.org/get-started/locally/) to install `PyTorch` first. GPU support is necessary.

After installing `PyTorch`, run

```bash
pip install transcript_transformer
```

## 📖 Quick Start

:::{tip}
:class: myclass1 myclass2
:name: fasta_predict
To apply TIS transformer on a small set of transcript sequences in fasta format, skip directly to [this section](#fa-process)
:::

Dictionary files (YAML/JSON) are the recommended approach to pass arguments to the tool. It is possible to list multiple configuration files. Required inputs are a **genome-level** reference and assembly file (`*.gtf`, `*.fa`).

In addition, when predicting translated ORFs through ribosome profiling data, TRISTAN requires ribosome profiling reads (`*.sam`/`*.bam`) **mapped to the transcriptome**

:::{tip}
:class: myclass1 myclass2
:name: tip_1
By default, most read-alignment tools function by outputting reads as aligned to the **genome**, and require additional directions to output reads aligned to the transcriptome. E.g., for [STAR](https://alexdobin.github.io/STAR/), the function flag `--TranscriptomeSAM` is required to output read alignments to the desired format.
:::


This can all be presented in a single configuration file `default.yml`:

```yaml
gtf_path : path/to/gtf_file.gtf
fa_path : path/to/fa_file.fa
########################################################
## add entries to ribosome profiling data.
## format: 'id : ribosome profiling paths'
########################################################
ribo_paths :
  SRR000001 : path/to/mapped/sample1.bam
  SRR000002 : path/to/mapped/sample2.bam
  SRR000003 : path/to/mapped/sample3.bam
########################################################
## database path (parsed data output)
########################################################
h5_path : my_experiment.h5
## Result prefix 
out_prefix: out/
```

TRISTAN tools perform the following steps:
1. Parse all data to a single HDF5 database (`h5_path`)
2. Finetune models on non-overlapping folds of the data.
3. Get model predictions for all positions of the transcriptome
4. Collect metadata for the top ranking predictions
5. Filter out [CDS variant calls](user_guide.md#CDS-filters)


#### TIS Transformer

TIS Transformer detects plausible coding sequences based on sequence context. TRISTAN incorporates pre-trained models for both the human and mouse organism. To evaluate all possible regions on the human genome, run:

```bash
tis_transformer default.yml --model human
```

This step will first create the database from the assembly files before advancing to the inference fase. To load in the data only, run the command with the `--data` flag.

Otherwise, to train a model from scratch, run:

```bash
tis_transformer default.yml
```


:::{attention}
:class: myclass1 myclass2
:name: tip-2
Training TIS Transformer from scratch is a very expensive process which can take multiple days. Make sure to check out the pre-trained models for the Human and Mouse organism before performing the training step. Otherwise, training should only be done once for each organism. Make sure to make a backup!
:::


#### RiboTIE

:::{tip}
:class: myclass1 myclass2
:name: tip-3
It is recommended to run TIS Transformer at least once on a newly create database. Predictions will be stored within the same `hdf5` database and incorporated in future outputs of RiboTIE. 
:::

RiboTIE detects actively translated ORFs based on ribosome profiling data. for every dataset (`ribo_paths`) listed, RiboTIE will fine-tune pre-trained models on non-overlapping folds of the data.

```bash
ribotie default.yml
```

:::{attention}
:class: myclass1 myclass2
:name: a-tip-reference
`SAM/BAM` files can require more than 100GB in RAM to load the data in. However, loading from the `hdf5` database is memory efficient. To first load in the data before utilizing the GPU, use `--data`.
:::

(fa-process)=
## 🧬 TIS Transformer: FA sequence

It is possible to apply TIS Transformer with only a fasta file containing transcript sequences to predict translated ORFs. This can be achieved using any of our pre-trained models. Here, `--fold` can be used to apply a model that was not trained on homologous transcript sequences listed in the input fasta file. If applicable, choose the model where the contig is featured within the test set as listed (here)[pretrained.md#tis-transformer]. For example, if the model features sequences that are transcript isoforms of a gene located on chromosome 1 of the human genome, we can run: 

```bash
tis_transformer --fasta <input.fa> --model human --fold 0 --accelerator cpu
```

Where `--accelerator cpu` tells the tool to use CPU resources (rather than a GPU).
