# How does it work? 

:::{attention}
:class: myclass1 myclass2
:name: how
Overall, this section is somewhat of a stub. Please refer to the original manuscripts of both tools for a deeper understanding on the methodology used.
:::

## 💾 Data loading
Information is separated by transcript and information type. Information belonging to a single transcript is mapped according to the index they populate within each `h5py.dataset`, used to store different types of information. [Variable length arrays](https://docs.h5py.org/en/stable/special.html#arbitrary-vlen-data) are used to store the sequences and annotations of all transcripts under a single data set. 
Sequences are stored using integer arrays following: `{A:0, T:1, C:2, G:3, N:4}`
An example `data.h5` has the following structure:


```
data.h5                                     (h5py.file)
    transcript                              (h5py.group)
    ├── tis                                 (h5py.dataset, dtype=vlen(int))
    ├── contig                              (h5py.dataset, dtype=str)
    ├── id                                  (h5py.dataset, dtype=str)
    ├── seq                                 (h5py.dataset, dtype=vlen(int))
    ├── ribo                                (h5py.group)
    │   ├── SRR0000001                      (h5py.group)
    │   │   ├── 5                           (h5py.group)
    │   │   │   ├── data                    (h5py.dataset, dtype=vlen(int))
    │   │   │   ├── indices                 (h5py.dataset, dtype=vlen(int))
    │   │   │   ├── indptr                  (h5py.dataset, dtype=vlen(int))
    │   │   │   ├── shape                   (h5py.dataset, dtype=vlen(int))
    │   ├── ...
    │   ....
    
```

Ribosome profiling data is saved by reads mapped to by **read length and transcript position**. Mapped reads are furthermore separated by their read lengths. As ribosome profiling data is often sparse, we made use of `scipy.sparse` to save data within the `h5` format. This allows us to save space and store matrix objects as separate arrays. Saving and loading of the data is achieved using the [``h5max``](https://github.com/jdcla/h5max) package.

<div align="center">
<img src="https://github.com/jdcla/h5max/raw/main/h5max.png" width="600">
</div>



## 🧮 RiboTIE

See [the manuscript](https://www.biorxiv.org/content/10.1101/2023.06.20.545724v1) for a detailed description.
RiboTIE detects translation initiation sites using only ribosome profiling data. The tool parses information on how reads are aligned along the transcript. Specifically, for each position, a vector containing the number of reads for each read length at that position is parsed.  **No sequence information is processed**. RiboTIE similarly returns predictions for each position on each transcript. From these predictions, ORFs are derived by a greedy search algorithm that finds the first in-frame stop codon. 

<div align="center">
<img src="https://github.com/TRISTAN-ORF/RiboTIE/raw/main/ribo_intro.png" width="800">
</div>

**Note:** striked-through text refers to steps typically performed by existing methods but ommited by RiboTIE.

Fine-tuning is important as ribosome profiling data has shown to be highly variable between experiments, with biological variability (tissues, sample age, ...) and technical variability (lab protocol, machine calibration, ...) playing a role.
RiboTIE is trained and fine-tuned using a set of canonical coding sequences. This approach does not prevent the trained model to find non-canonical ORFs as the model is trained on a different set of transcripts it is applied on for prediction.
After fine-tuning, the model provides predictions for each  position of each transcript. 
From these predicted translation initiation sites, the resulting translated ORFs are obtained by searching for the first in-frame stop codon.
Filtering can be applied based on the characteristics of the translated ORFs (e.g. start codon, minimum length).

This technique was shown to substantially outperform previous methods. We hypothesize this gain to be achieved through various factors:
- fine-tuning on each data set, the model learns custom rules present for each data set
- inclusion of read length information
- very few custom hardcoded rules for data (pre-)processing or selection.
- use of a state-of-the-art machine learning tools (transformer networks), which are perfectly suited for the data type (a variable number of input vectors). 


## ⁉️ FAQ

**Will training and selecting models on canonical coding sequences limit the ability of RiboTIE to detect non-canonical ORFs?**

> RiboTIE has been trained only on ribosome sequencing data and non-overlapping folds to prevent the model from relying on ORF metadata or (start) codon information, from which the length of ORFs is easily derived. As such, RiboTIE would only be biased against detecting TISs of shorter ORFs if the distribution of ribosome protected fragments is distinctly different from that of canonincal coding sequences, for which there exists no proof today.  This statement is reflected by our benchmark of RiboTIE against other existing tools, where the discovery rate of short canonical CDSs (<300nt) by RiboTIE was at least 400% higher than any other tool [RiboTie paper; Fig 1C](http://biorxiv.org/cgi/content/full/2024.03.21.586110v1). 

**Why not create a model that is trained on both transcript sequence and ribosome profiling data?**

> Models trained on sequencing data are susceptible to incorporating biases existent in today's annotations (e.g. codon usage, start codons frequency, ...). On the other hand, models applying ribosome profiling data are restricted by the read depth of the sample and translation profile of the organism. Both models, in essence, are optimized to answer different questions (i.e., What ORFs look viable to be translated based on sequence context / What ORFs are actively being translated based on ribosome protected fragment distributions). We determined that the combination of both modalities of data within a single model obfuscates the function of that model.

> In practice, these differences become a problem when evaluating the optimization process. For TIS transformer, all transcripts are viable information, as the sequence of all transcripts is known. For RiboTIE, only transcripts with mapped reads have valuable information. When optimizing RiboTIE, we do not alter the labels of the positive set for the transcripts with a low number of mapped reads (which would require us to solve the question of how many reads are too few to alter target labels to a negative value) as the presence of a positive label for an input vector containing zeros (no reads) does not bias the model negatively (there is nothing to learn). However, combining both sequence and ribo-seq data would result in the model relying on sequence information more than it does on ribosome profiling data.

> In conclusion, the combination of both data modalities raises plenty of hard-to-answer questions and fails to offer a clear understanding on the actual function performed by the optimized model. Both tools can be used in combination to offer insights into ORFs of interest, but should be evaluated using a good understanding of the advantages and sensitivies of both approaches.