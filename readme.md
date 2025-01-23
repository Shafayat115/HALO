# Fast-Part

**Fast-Part** is an efficient tool for partitioning and clustering sequences. It uses CD-HIT or MMseqs2 for clustering, along with DIAMOND for alignments, to enable streamlined handling of biological sequences, automated partitioning into training and test sets, and iterative reassignment based on DIAMOND alignments. 

## Features

- **Clustering Options**: Supports clustering with **CD-HIT** or **MMseqs2**.
- **Automatic Word Length Adjustment**: For CD-HIT, automatically adjusts word length based on the identity threshold.
- **Default to MMseqs2**: Automatically defaults to MMseqs2 if the identity threshold is set below 0.4.
- **Train-Test Partitioning**: Automatically applies a 5% adjustment to the specified train ratio.
- **Detailed Summary**: Outputs a summary file with configuration details, partition sizes, removed sequences, and execution time.

## Requirements

- **Python 3.6+**
- **CD-HIT**
- **MMseqs2**
- **DIAMOND**
- **Biopython** library (`pip install biopython`)

Ensure that **CD-HIT**, **MMseqs2**, and **DIAMOND** are installed and accessible in your system’s PATH.

## Installation Instructions

Before using **Fast_Part**, ensure that **CD-HIT**, **MMseqs2**, and **DIAMOND** are installed on your system. Below are installation instructions for each tool.

### CD-HIT

CD-HIT is a clustering tool for protein and nucleotide sequences.

1. **Download CD-HIT**:
   - Go to the [CD-HIT website](https://github.com/weizhongli/cdhit) and download the latest release.
   
2. **Install CD-HIT**:
   - Extract the downloaded file and navigate to the extracted folder.
   - Run the following commands:
     ```bash
     make
     sudo make install
     ```
   - Ensure the `cd-hit` binary is accessible in your PATH. You can test it with:
     ```bash
     cd-hit --help
     ```
