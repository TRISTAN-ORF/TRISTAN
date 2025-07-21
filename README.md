<div align="center">
<h1>TRISTAN</h1> 

*Driving coding sequence discovery since 2023*

[![PyPi Version](https://img.shields.io/pypi/v/transcript-transformer.svg)](https://pypi.python.org/pypi/transcript-transformer/)
[![GitHub license](https://img.shields.io/github/license/TRISTAN-ORF/transcript_transformer)](https://github.com/TRISTAN-ORF/transcript_transformer/blob/main/LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/TRISTAN-ORF/transcript_transformer)](https://github.com/TRISTAN-ORF/transcript_transformer/issues)
[![GitHub stars](https://img.shields.io/github/stars/TRISTAN-ORF/transcript_transformer)](https://github.com/TRISTAN-ORF/transcript_transformer/stargazers)
[![Documentation Status](https://readthedocs.org/projects/tristan-orf/badge/?version=latest)](https://tristan-orf.readthedocs.io/en/latest/?badge=latest)
</div>

**TRISTAN** (TRanslational Identification Suite using Transformer Networks for ANalysis) is a suite of tools for detecting translated Open Reading Frames (ORFs) in organisms through the analysis of sequence context and/or ribosome profiling (Ribo-seq) data.

---

### 📚 Documentation

**For complete installation instructions, user guides, and tutorials, please see our full documentation on [ReadTheDocs](https://transcript-transformer.readthedocs.io/en/latest/).**

---

## 👋 About the Project

TRISTAN tools are built using the newest advances and best practices in machine learning, stepping away from manually curated rules for data processing and instead letting the optimization algorithm handle it. The tools are designed with flexibility and modularity in mind.

Key design principles:

*   **Unbiased Data Utilization**: Leverages the full transcriptome, allowing models to learn complex translational patterns directly from biological data without pre-imposed biases.
*   **Robust Model Validation**: Separates training, validation, and test sets by chromosome to prevent information leakage and provide a more accurate assessment of performance.
*   **Data-Driven Decision Making**: Machine learning models learn nuances intrinsically, avoiding hardcoded rules for data alteration or prediction adjustments.
*   **Seamless Downstream Integration**: Generates various output file formats (CSV, GTF) designed for easy integration with common downstream analysis tools.

The `transcript-transformer` package incorporates the functionality of **TIS Transformer** and **RiboTIE**, using the **Performer** architecture to process transcripts at single-nucleotide resolution.

## 🛠️ Installation

First, install `PyTorch` with GPU support by following the instructions using [the PyTorch manual](https://pytorch.org/get-started/locally/).

Then, install TRISTAN:
```bash
pip install transcript_transformer
```

## 📖 Quick Start

TRISTAN can be used to detect translated ORFs using an input fasta file. See the (User Documentation)[https://transcript-transformer.readthedocs.io/en/latest/getting_started.html#tis-transformer-fa-sequence] for more info.

Otherwise, TRISTAN is run from the command line, using a YAML configuration file to specify inputs and parameters.

1.  **Create a configuration file (`config.yml`):**

    ```yaml
    # Path to genome annotation and sequence
    gtf_path : path/to/gtf_file.gtf
    fa_path : path/to/fa_file.fa
    
    # Path for the HDF5 database
    h5_path : my_experiment.h5
    
    # Prefix for output files
    out_prefix: out/
    
    # (Optional) Add ribosome profiling data for RiboTIE
    ribo_paths :
      SRR000001 : path/to/mapped/sample1.bam
      SRR000002 : path/to/mapped/sample2.bam
    ```

2.  **Run the tools:**

    *   **TIS Transformer:** Detect ORFs based on sequence context. Pre-trained models for human and mouse are available.

        ```bash
        # Use a pre-trained model for human
        tis_transformer config.yml --model human
        ```

    *   **RiboTIE:** Detect actively translated ORFs from Ribo-seq data.

        ```bash
        # Fine-tune and predict from Ribo-seq samples
        ribotie config.yml
        ```

    > For more advanced usage, including training models from scratch and parallel processing, please refer to the full documentation.

## 🖊️ How to Cite

If you use TRISTAN in your research, please cite the relevant papers:

**TIS Transformer:**
> Clauwaert, J., McVey, Z., Gupta, R., & Menschaert, G. (2023). TIS Transformer: remapping the human proteome using deep learning. *NAR Genomics and Bioinformatics*, 5(1), lqad021. https://doi.org/10.1093/nargab/lqad021

```bibtex
@article {10.1093/nargab/lqad021,
    author = {Clauwaert, Jim and McVey, Zahra and Gupta, Ramneek and Menschaert, Gerben},
    title = "{TIS Transformer: remapping the human proteome using deep learning}",
    journal = {NAR Genomics and Bioinformatics},
    volume = {5},
    number = {1},
    year = {2023},
    month = {03},
    issn = {2631-9268},
    doi = {10.1093/nargab/lqad021}
}
```

**RiboTIE:**
> Clauwaert, J., et al. (2025). Deep learning to decode sites of RNA translation in normal and cancerous tissues. *Nature Communications*, 16(1), 1275. https://doi.org/10.1038/s41467-025-56543-0

```bibtex
@article{clauwaert2025deep,
  title={Deep learning to decode sites of RNA translation in normal and cancerous tissues},
  author={Clauwaert, Jim and McVey, Zahra and Gupta, Ramneek and Yannuzzi, Ian and Basrur, Venkatesha and Nesvizhskii, Alexey I and Menschaert, Gerben and Prensner, John R},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={1275},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
