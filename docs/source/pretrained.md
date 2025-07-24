# Available models

TRISTAN has several pre-trained models available that allow direct prediction of transcript features. Models are always trained and applied on non-overlapping folds of the data to prevent overfitting and bias in the prediction results. When applying TRISTAN models for your own data, make sure the contig names assigned to the different folds align with those used in your input files. See [the user guide on saved model configs](user_guide.md#trained-models) to understand how to interpret model config files and how to alter contig allocations. 

(tis-transformer)=
## 🧬 TIS Transformer

### Human (Ensembl v113)
Applied when using `--model human` for the `tis_transformer` tool. 

Training data:
- [Fasta](https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/)
- [GTF](https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/)


Models (Folds):
:::{tip}
:class: myclass1 myclass2
:name: wild-list
`[]` denotes a wild card for lists, any contig not recognized in the Train and Validation set will be populated here.
:::

0.  **File:** `Homo_sapiens.GRCh38.113_f0.tt.ckpt`
    * **Test Set:** `[]`
    * **Train Set:** `10`, `11`, `12`, `13`, `15`, `17`, `18`, `19`, `20`, `21`, `3`, `5`, `6`, `7`, `8`, `GL000008.2`, `GL000195.1`, `GL000205.2`, `GL000214.1`, `GL000218.1`, `GL000220.1`, `GL000221.1`, `GL000224.1`, `KI270710.1`, `KI270711.1`, `KI270718.1`, `KI270721.1`, `KI270722.1`, `KI270726.1`, `KI270727.1`, `KI270728.1`, `KI270731.1`, `KI270733.1`, `KI270734.1`, `KI270741.1`, `KI270742.1`, `KI270743.1`, `KI270744.1`, `KI270748.1`, `KI270749.1`, `KI270751.1`, `KI270753.1`, `X`
    * **Validation Set:** `14`, `2`, `9`, `GL000009.2`, `GL000194.1`, `GL000213.1`, `GL000216.2`, `GL000219.1`, `GL000225.1`, `KI270442.1`, `KI270706.1`, `KI270712.1`, `KI270713.1`, `KI270714.1`, `KI270717.1`, `KI270719.1`, `KI270720.1`, `KI270745.1`, `KI270746.1`, `KI270750.1`, `KI270755.1`, `MT`, `Y`
1.  **File:** `Homo_sapiens.GRCh38.113_f1.tt.ckpt`
    * **Test Set:** `2`, `20`, `5`, `8`, `GL000008.2`, `GL000009.2`, `GL000195.1`, `GL000205.2`, `GL000213.1`, `GL000216.2`, `GL000218.1`, `GL000219.1`, `GL000221.1`, `KI270442.1`, `KI270706.1`, `KI270712.1`, `KI270713.1`, `KI270714.1`, `KI270717.1`, `KI270719.1`, `KI270720.1`, `KI270726.1`, `KI270728.1`, `KI270731.1`, `KI270742.1`, `KI270748.1`, `KI270749.1`, `KI270750.1`, `KI270751.1`, `KI270755.1`, `MT`, `Y`
    * **Train Set:** `1`, `10`, `11`, `13`, `14`, `15`, `17`, `18`, `19`, `22`, `3`, `4`, `6`, `9`, `X`
    * **Validation Set:** `12`, `16`, `21`, `7`, `GL000194.1`, `GL000214.1`, `GL000220.1`, `GL000224.1`, `GL000225.1`, `KI270710.1`, `KI270711.1`, `KI270718.1`, `KI270721.1`, `KI270722.1`, `KI270727.1`, `KI270733.1`, `KI270734.1`, `KI270741.1`, `KI270743.1`, `KI270744.1`, `KI270745.1`, `KI270746.1`, `KI270753.1`
2.  **File:** `Homo_sapiens.GRCh38.113_f2.tt.ckpt`
    * **Test Set:** `10`, `14`, `17`, `21`, `3`, `GL000194.1`, `GL000214.1`, `GL000220.1`, `GL000224.1`, `GL000225.1`, `KI270710.1`, `KI270711.1`, `KI270718.1`, `KI270721.1`, `KI270722.1`, `KI270727.1`, `KI270733.1`, `KI270734.1`, `KI270741.1`, `KI270743.1`, `KI270744.1`, `KI270745.1`, `KI270746.1`, `KI270753.1`
    * **Train Set:** `1`, `12`, `16`, `18`, `19`, `2`, `20`, `22`, `4`, `6`, `7`, `8`, `9`, `GL000008.2`, `GL000009.2`, `GL000195.1`, `GL000205.2`, `GL000213.1`, `GL000218.1`, `GL000219.1`, `GL000221.1`, `KI270712.1`, `KI270713.1`, `KI270714.1`, `KI270717.1`, `KI270719.1`, `KI270728.1`, `KI270731.1`, `KI270742.1`, `KI270748.1`, `KI270749.1`, `KI270751.1`, `KI270755.1`, `X`, `Y`
    * **Validation Set:** `11`, `13`, `15`, `5`, `GL000216.2`, `KI270442.1`, `KI270706.1`, `KI270720.1`, `KI270726.1`, `KI270750.1`, `MT`
3.  **File:** `Homo_sapiens.GRCh38.113_f3.tt.ckpt`
    * **Test Set:** `12`, `13`, `19`, `6`, `X`
    * **Train Set:** `1`, `10`, `11`, `15`, `16`, `17`, `18`, `20`, `21`, `22`, `3`, `4`, `5`, `7`, `8`, `GL000008.2`, `GL000009.2`, `GL000194.1`, `GL000195.1`, `GL000205.2`, `GL000214.1`, `GL000218.1`, `GL000219.1`, `GL000221.1`, `GL000224.1`, `KI270442.1`, `KI270706.1`, `KI270711.1`, `KI270712.1`, `KI270714.1`, `KI270717.1`, `KI270718.1`, `KI270719.1`, `KI270721.1`, `KI270726.1`, `KI270728.1`, `KI270733.1`, `KI270734.1`, `KI270741.1`, `KI270742.1`, `KI270743.1`, `KI270744.1`, `KI270745.1`, `KI270748.1`, `KI270749.1`, `KI270750.1`, `KI270753.1`, `KI270755.1`
    * **Validation Set:** `14`, `2`, `9`, `GL000213.1`, `GL000216.2`, `GL000220.1`, `GL000225.1`, `KI270710.1`, `KI270713.1`, `KI270720.1`, `KI270722.1`, `KI270727.1`, `KI270731.1`, `KI270746.1`, `KI270751.1`, `MT`, `Y`
4.  **File:** `Homo_sapiens.GRCh38.113_f4.tt.ckpt`
    * **Test Set:** `11`, `15`, `18`, `7`, `9`
    * **Train Set:** `1`, `10`, `12`, `13`, `14`, `17`, `19`, `2`, `20`, `22`, `3`, `4`, `8`, `GL000008.2`, `GL000009.2`, `GL000220.1`, `GL000221.1`, `GL000224.1`, `GL000225.1`, `KI270706.1`, `KI270710.1`, `KI270712.1`, `KI270713.1`, `KI270717.1`, `KI270719.1`, `KI270720.1`, `KI270722.1`, `KI270726.1`, `KI270734.1`, `KI270744.1`, `KI270749.1`, `KI270750.1`, `KI270751.1`, `KI270753.1`, `X`, `Y`
    * **Validation Set:** `16`, `21`, `5`, `6`, `GL000194.1`, `GL000195.1`, `GL000205.2`, `GL000213.1`, `GL000214.1`, `GL000216.2`, `GL000218.1`, `GL000219.1`, `KI270442.1`, `KI270711.1`, `KI270714.1`, `KI270718.1`, `KI270721.1`, `KI270727.1`, `KI270728.1`, `KI270731.1`, `KI270733.1`, `KI270741.1`, `KI270742.1`, `KI270743.1`, `KI270745.1`, `KI270746.1`, `KI270748.1`, `KI270755.1`, `MT`


### Mouse (Ensembl v112)

Applied when using `--model mouse` for the `tis_transformer` tool. 

Training data:
- [Fasta](https://ftp.ensembl.org/pub/release-112/fasta/mus_musculus/)
- [GTF](https://ftp.ensembl.org/pub/release-112/gtf/mus_musculus/)

0.  **File:** `Mus_musculus.GRCm39.112_f0.tt.ckpt`
    * **Test Set:** `[]`
    * **Train Set:** `1`, `10`, `11`, `12`, `14`, `16`, `17`, `19`, `2`, `4`, `5`, `8`, `X`, `Y`
    * **Validation Set:** `13`, `6`, `9`, `GL456210.1`, `GL456211.1`, `GL456212.1`, `GL456219.1`, `GL456221.1`, `GL456239.1`, `GL456354.1`, `GL456372.1`, `GL456381.1`, `GL456385.1`, `JH584296.1`, `JH584297.1`, `JH584298.1`, `JH584299.1`, `JH584303.1`, `JH584304.1`, `MT`
1.  **File:** `Mus_musculus.GRCm39.112_f1.tt.ckpt`
    * **Test Set:** `14`, `19`, `2`, `4`
    * **Train Set:** `1`, `10`, `11`, `12`, `15`, `16`, `17`, `18`, `3`, `5`, `7`, `8`, `X`, `Y`
    * **Validation Set:** `13`, `6`, `9`, `GL456210.1`, `GL456211.1`, `GL456212.1`, `GL456219.1`, `GL456221.1`, `GL456239.1`, `GL456354.1`, `GL456372.1`, `GL456381.1`, `GL456385.1`, `JH584296.1`, `JH584297.1`, `JH584298.1`, `JH584299.1`, `JH584303.1`, `JH584304.1`, `MT`
2.  **File:** `Mus_musculus.GRCm39.112_f2.tt.ckpt`
    * **Test Set:** `1`, `17`, `8`, `X`
    * **Train Set:** `10`, `11`, `13`, `14`, `15`, `16`, `18`, `19`, `2`, `4`, `6`, `7`, `9`, `Y`
    * **Validation Set:** `12`, `3`, `5`, `GL456210.1`, `GL456211.1`, `GL456212.1`, `GL456219.1`, `GL456221.1`, `GL456239.1`, `GL456354.1`, `GL456372.1`, `GL456381.1`, `GL456385.1`, `JH584296.1`, `JH584297.1`, `JH584298.1`, `JH584299.1`, `JH584303.1`, `JH584304.1`, `MT`
3.  **File:** `Mus_musculus.GRCm39.112_f3.tt.ckpt`
    * **Test Set:** `12`, `13`, `5`, `6`, `Y`
    * **Train Set:** `1`, `10`, `14`, `16`, `17`, `18`, `19`, `2`, `3`, `7`, `8`, `9`, `X`
    * **Validation Set:** `11`, `15`, `4`, `GL456210.1`, `GL456211.1`, `GL456212.1`, `GL456219.1`, `GL456221.1`, `GL456239.1`, `GL456354.1`, `GL456372.1`, `GL456381.1`, `GL456385.1`, `JH584296.1`, `JH584297.1`, `JH584298.1`, `JH584299.1`, `JH584303.1`, `JH584304.1`, `MT`
4.  **File:** `Mus_musculus.GRCm39.112_f4.tt.ckpt`
    * **Test Set:** `10`, `11`, `16`, `9`, `GL456210.1`, `GL456211.1`, `GL456212.1`, `GL456219.1`, `GL456221.1`, `GL456239.1`, `GL456354.1`, `GL456372.1`, `GL456381.1`, `GL456385.1`, `JH584296.1`, `JH584297.1`, `JH584298.1`, `JH584299.1`, `JH584303.1`, `JH584304.1`, `MT`
    * **Train Set:** `13`, `14`, `15`, `17`, `18`, `19`, `2`, `4`, `5`, `6`, `7`, `8`, `X`, `Y`
    * **Validation Set:** `1`, `12`, `3`



## 🧮 RiboTIE

RiboTIE is pre-trained on a variety of datasets to kickstart a basic understanding of ribosome read profiles along the transcriptome. **The pre-trained model can be applied to any organism**, where it is very unlikely to suffer from overfitting as the model does not process any sequencing data.

Training data:
- [Fasta (GRCh38v110)](https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/)
- [GTF (GRCh38v110)](https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/)
- Ribo-seq data: `SRR592960`, `SRR1562539`, `SRR1573939`, `SRR1610244`, `SRR1976443`, `SRR2536856`, `SRR2873532`, `SRR3575904`.


0.  **File:** `50perc_06_23_f0.rt.ckpt`
    * **Test Set:** `[]`
    * **Train Set:** `3`, `5`, `7`, `11`, `13`, `15`, `19`, `21`, `X`, `chr3`, `chr5`, `chr7`, `chr11`, `chr13`, `chr15`, `chr19`, `chr21`, `chrX`
    * **Validation Set:** `1`, `9`, `17`, `chr1`, `chr9`, `chr17`
1.  **File:** `50perc_06_23_f1.rt.ckpt`
    * **Test Set:** `1`, `3`, `5`, `7`, `9`, `11`, `13`, `15`, `17`, `19`, `21`, `X`, `chr1`, `chr3`, `chr5`, `chr7`, `chr9`, `chr11`, `chr13`, `chr15`, `chr17`, `chr19`, `chr21`, `chrX`
    * **Train Set:** `2`, `6`, `8`, `10`, `14`, `16`, `18`, `22`, `Y`, `chr2`, `chr6`, `chr8`, `chr10`, `chr14`, `chr16`, `chr18`, `chr22`, `chrY`
    * **Validation Set:** `4`, `12`, `20`, `chr4`, `chr12`, `chr20`
