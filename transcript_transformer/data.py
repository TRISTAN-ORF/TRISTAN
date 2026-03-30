import os
import traceback
import logging
from copy import deepcopy
import shutil
import time
import numpy as np

from scipy import sparse
from tqdm import tqdm
import polars as pl

import h5py
import h5max
import pyfaidx
import pyranges as pr
from .util_functions import (
    vec2DNA,
    construct_prot,
    slice_gen,
    prot2vec,
    check_genomic_order,
    prtime,
    derive_exon_number
)
from transcript_transformer import (
    REQ_HEADERS,
    CUSTOM_HEADERS,
    DROPPED_HEADERS,
    COMPAT_MAPPING,
)


def process_seq_data(h5_path, gtf_path, fa_path, backup_path, backup=True):
    prtime("Loading in assembly data...", "\n")
    pulled = False
    if not backup_path:
        backup_path = os.path.splitext(gtf_path)[0] + ".h5"
    if os.path.abspath(backup_path) == os.path.abspath(h5_path):
        print(
            f"\t !- Backup path identical to h5 output path, no database copy will be created..."
        )
        backup = False
    elif not os.path.isfile(h5_path) and os.path.isfile(backup_path):
        print(f"\t -- Processed assembly data restored ({backup_path})")
        shutil.copy(backup_path, h5_path)
        pulled = True
    if os.path.isfile(h5_path):
        f = h5py.File(h5_path, "r")
        if "transcript" in f.keys():
            print(
                "\t -- Parsed transcriptome directory found, "
                "assembly information can not be re-processed (for existing h5 files)."
            )
        f.close()
    else:
        DNA_seq = pyfaidx.Fasta(fa_path)
        gr = pr.read_gtf(gtf_path)
        gtf = pl.from_pandas(gr.df).rename(COMPAT_MAPPING)
        # pyranges converts GTF 1-based Start to 0-based; restore to 1-based GTF convention
        gtf = gtf.with_columns(pl.col("start") + 1)
        # import exon number as int (strings have wrong sortin (e.g. 10, 11, 2,...))
        if "exon_number" not in gtf.columns:
            gtf = derive_exon_number(gtf)
        gtf = gtf.with_columns(pl.col("exon_number").cast(pl.Int32, strict=False))
        db_tr = parse_transcriptome(gtf, DNA_seq)
        db_gtf = parse_genome(gtf)
        no_handle = True
        max_wait = 900
        waited = 0
        while no_handle and (waited < max_wait):
            try:
                f = h5py.File(h5_path, "a")
                no_handle = False
                try:
                    print(f"\t -- Saving transcriptome data to {h5_path}...")
                    f = save_transcriptome_to_h5(f, db_tr)
                    if len(db_gtf) > 0:
                        print(f"\t -- Saving genome data to {h5_path}...")
                        f = save_genome_to_h5(f, db_gtf)
                    f.close()
                    if backup and (not pulled):
                        shutil.copy(h5_path, backup_path)
                except Exception as e:
                    logging.error(traceback.format_exc())
                    print("Failed to update h5 database, which might be corrupted")
            except Exception as e:
                if waited < max_wait:
                    time.sleep(120)
                    waited += 120
        if no_handle:
            print("Could not open h5 database, suspending...")


def process_ribo_data(
    h5_path,
    ribo_paths,
    overwrite=False,
    parallel=False,
    low_memory=False,
):
    prtime("Loading in Ribo-seq data...", "\n")
    # TODO implement option to run custom read lens
    read_lims = [20, 41]
    # load from hdf5 file
    f = h5py.File(h5_path, "r")
    tr_ids = pl.Series(np.array(f["transcript/transcript_id"]), dtype=pl.Utf8)
    tr_lens = pl.from_numpy(np.array(f["transcript/transcript_len"])).to_series()
    samples = {}
    for group_samples in ribo_paths.values():
        samples.update(group_samples)
    samples_to_process = deepcopy(samples)
    for sample_id, path in samples.items():
        cond_1 = (
            parallel
            and (not overwrite)
            and (os.path.isfile(h5_path.split(".h5")[0] + f"_{sample_id}.h5"))
        )
        cond_2 = not (parallel or overwrite) and (
            f"transcript/riboseq/{sample_id}" in f.keys()
        )
        if cond_1 or cond_2:
            print(
                f"\t -- {sample_id} in h5, omitting..."
                "(use --overwrite_data for overwriting existing riboseq data)"
            )
            samples_to_process.pop(sample_id)
    f.close()
    for sample_id, path in samples_to_process.items():
        prtime(f"Loading in {sample_id}...", "\n")
        try:
            if not parallel:
                print(f"\t -- Saving data to {h5_path}...")
                f = h5py.File(h5_path, "a")
            else:
                path_h5 = h5_path.split(".h5")[0] + f"_{sample_id}.h5"
                print(f"\t -- Saving data to {path_h5}...")
                f = h5py.File(path_h5, "w")
                f.create_group("transcript")
                # Save tr_ids in parallel h5 files
                max_char_len = tr_ids.str.len_chars().max()
                f["transcript"].create_dataset(
                    "transcript_id", data=tr_ids, dtype=f"<S{max_char_len}"
                )
            if "riboseq" not in f["transcript"].keys():
                f["transcript"].create_group("riboseq")
            if sample_id in f["transcript/riboseq"].keys():
                del f[f"transcript/riboseq/{sample_id}"]
            f["transcript/riboseq"].create_group(sample_id)
            exp_grp = f[f"transcript/riboseq/{sample_id}"].create_group("5")
            
            num_reads = parse_ribo_reads(path, read_lims, tr_ids, tr_lens, exp_grp, low_memory)
            
            exp_grp.create_dataset("num_reads", data=np.array(num_reads).astype(int))
            exp_grp.create_dataset("metadata", data=read_lims)
            f.close()
        except Exception as error:
            print(error)
            print(traceback.format_exc())
            if 'f' in locals() and sample_id in f["transcript/riboseq"].keys():
                del f[f"transcript/riboseq/{sample_id}"]
            if 'f' in locals():
                f.close()


def save_genome_to_h5(f, db):
    grp = f.create_group("gene")
    for key in db.columns:
        if db[key].dtype == pl.Categorical:
            db = db.with_columns(pl.col(key).cast(pl.String))
        if db[key].dtype == pl.String:
            array = [a if a != None else "" for a in db[key]]
            max_char_len = db[key].str.len_chars().max()
            if max_char_len > 0:
                grp.create_dataset(key, data=array, dtype=f"<S{max_char_len}")
            else:
                continue
        else:
            grp.create_dataset(key, data=db[key])

    return f


def save_transcriptome_to_h5(f, db):
    dt8 = h5py.vlen_dtype(np.dtype("int8"))
    dt = h5py.vlen_dtype(np.dtype("int"))
    grp = f.create_group("transcript")
    for key in db.columns:
        if db[key].dtype == pl.Categorical:
            db = db.with_columns(pl.col(key).cast(pl.String))
        if db[key].dtype == pl.String:
            array = [a if a != None else "" for a in db[key]]
            max_char_len = db[key].str.len_chars().max()
            if (max_char_len is not None) & (max_char_len > 0):
                grp.create_dataset(key, data=array, dtype=f"<S{max_char_len}")
            else:
                continue
        elif key in ["seq", "tis", "canonical_protein_seq"]:
            grp.create_dataset(key, data=db[key], dtype=dt8)
        elif key in ["exon_idxs", "exon_coords", "CDS_idxs", "CDS_coords"]:
            grp.create_dataset(key, data=np.array(db[key], dtype=object), dtype=dt)
        else:
            grp.create_dataset(key, data=db[key])

    return f


def parse_genome(gtf):
    gene_gtf = gtf.filter(pl.col("feature") == "gene")
    cols_to_drop = ["score", "frame"]
    for col in gene_gtf.columns[8:]:
        if gene_gtf.schema[col] == pl.Float64:
            if gene_gtf[col].null_count() == gene_gtf.height or all(
                gene_gtf[col].is_nan()
            ):
                cols_to_drop.append(col)
        else:
            if gene_gtf[col].null_count() == gene_gtf.height:
                cols_to_drop.append(col)

    gene_gtf = gene_gtf.drop(cols_to_drop)

    return gene_gtf


def parse_transcriptome(gtf, DNA_seq):
    # ensure all required fields are listed
    assert np.isin(
        REQ_HEADERS, gtf.columns
    ).all(), f"Not all required properties in gtf file: {REQ_HEADERS}"
    # evaluate extra columns
    xtr_cols = np.array(gtf.columns)[
        ~pl.Series(gtf.columns).is_in(REQ_HEADERS).to_numpy()
    ]
    data_dict_keys = np.array(REQ_HEADERS + CUSTOM_HEADERS + list(xtr_cols))
    data_dict = {k: [] for k in CUSTOM_HEADERS}

    print("\t -- Reading in assembly...")
    gtf_set = gtf.filter(
        # exclude transcript ids that are empty
        pl.col("transcript_id") != "",
        pl.col("feature").is_in(
            ["transcript", "exon", "CDS", "start_codon", "stop_codon"]
        ),
    ).sort(["seqname", "transcript_id", "exon_number"])

    gtf_set = gtf_set.with_columns(
        (abs(pl.col("start") - pl.col("end")) + 1).alias("feature_length")
    )
    trs = gtf_set["transcript_id"].unique(maintain_order=True)

    db = pl.DataFrame(data={"transcript_id": trs})
    db = db.join(gtf.filter(pl.col("feature") == "transcript"), on="transcript_id")

    altered_tr_exons = []
    print("\t -- Importing transcripts and metadata...")
    for tr_id, gtf_tr in tqdm(
        gtf_set.group_by("transcript_id", maintain_order=True), total=len(db)
    ):
        is_pos_strand = (gtf_tr["strand"] == "+").any()
        # assert start > end
        assert any(
            gtf_tr["start"] <= gtf_tr["end"]
        ), f"Start and end coordinates are not correct for transcript {tr_id}"
        # Check and fix exon ordering
        gtf_tmp = gtf_tr.filter(pl.col("feature") == "exon").sort(
            "start", descending=[not is_pos_strand]
        )
        gtf_tmp = gtf_tmp.with_columns(
            exon_number_alt=pl.Series(np.arange(1, gtf_tmp.height + 1))
        )
        if any(gtf_tmp["exon_number"] != gtf_tmp["exon_number_alt"]):
            exon_dict = dict(
                gtf_tmp.select(["exon_number", "exon_number_alt"]).iter_rows()
            )
            gtf_tr = gtf_tr.with_columns(pl.col("exon_number").replace(exon_dict)).sort(
                "exon_number"
            )
            altered_tr_exons.append(tr_id)
        ftrs = {}
        ftr_cum_lens = {}
        ftr_idxs = {}
        for feature, feature_df in gtf_tr.group_by("feature", maintain_order=True):
            ftrs[feature[0]] = feature_df
            ftr_lens = feature_df["feature_length"].drop_nulls().to_numpy()
            cum_lens = np.insert(np.cumsum(ftr_lens), 0, 0)
            ftr_cum_lens[feature[0]] = cum_lens
            # feature boundaries; tuples flattened into single vector (e.g. [0,10,10,12,12,20])
            ftr_idxs[feature[0]] = np.vstack((cum_lens[:-1], cum_lens[1:])).T.ravel()

        data_dict["transcript_len"].append(ftr_cum_lens["exon"].max())
        if ftr_cum_lens["exon"].max() == 0:
            print(
                "WARNING: No exons found for transcript. This should not happen. Please ensure"
                "exons are marked with the correct transcript id"
            )
        # TODO: when multiple TISs are supported, code needs update
        # init empty boolean to denote TIS locations
        target_seq = np.full(ftr_cum_lens["exon"].max(), False)

        exon_coords = []
        exon_seqs = []
        for exon_i, exon in enumerate(ftrs["exon"].iter_rows(named=True)):
            # get sequence
            exon_seq = slice_gen(
                DNA_seq[exon["seqname"]],
                exon["start"],
                exon["end"],
                exon["strand"],
                to_vec=True,
            ).astype(np.int16)
            exon_coords.append(exon["start"])
            exon_coords.append(exon["end"])
            exon_seqs.append(exon_seq)
        seq = np.concatenate(exon_seqs)

        if "CDS" in ftrs:
            # select first in case of split (intron) start codon
            first_cds = ftrs["CDS"][0].to_dicts()[0]
            exon_i = first_cds["exon_number"] - 1
            exon = ftrs["exon"][exon_i].to_dicts()[0]
            # shift CDS transcript idxs based on start exon
            exon_shift = ftr_cum_lens["exon"][exon_i]
            if is_pos_strand:
                # shift CDS transcript idxs based on cds start in exon
                in_exon_shift = ftrs["CDS"][0, "start"] - exon["start"]
                tis = first_cds["start"]
                tis_idx = ftr_cum_lens["exon"][exon_i] + tis - exon["start"]
                lts = ftrs["CDS"][-1].to_dicts()[0]["end"]
                if "stop_codon" in ftrs:
                    tts = ftrs["stop_codon"][0][0, "start"]
                else:
                    tts = -1
            else:
                # shift CDS transcript idxs based on cds start in exon
                in_exon_shift = exon["end"] - ftrs["CDS"][0, "end"]
                tis = first_cds["end"]
                tis_idx = ftr_cum_lens["exon"][exon_i] + exon["end"] - tis
                lts = ftrs["CDS"][-1].to_dicts()[0]["start"]
                if "stop_codon" in ftrs:
                    tts = ftrs["stop_codon"][0][0, "end"]
                else:
                    tts = -1
            target_seq[tis_idx] = 1
            DNA_frag = vec2DNA(seq[tis_idx:])
            prot, _, _ = construct_prot(DNA_frag)
            data_dict["has_annotated_stop_codon"].append("stop_codon" in ftrs)
            data_dict["has_annotated_start_codon"].append("start_codon" in ftrs)
            data_dict["CDS_idxs"].append(ftr_idxs["CDS"] + exon_shift + in_exon_shift)
            CDS_coords = (
                ftrs["CDS"][:, ["start", "end"]]
                .transpose()
                .unpivot()["value"]
                .to_numpy()
            )
            check_genomic_order(CDS_coords, "+" if is_pos_strand else "-")
            data_dict["CDS_coords"].append(CDS_coords)
            data_dict["canonical_TIS_exon"].append(exon_i + 1)
            data_dict["canonical_TIS_idx"].append(tis_idx)
            # LTS: Last Translation Site; 1 nucleotide upstream of TTS
            tts_idx = tis_idx + ftr_cum_lens["CDS"].max()
            data_dict["canonical_TTS_idx"].append(tts_idx)
            data_dict["canonical_LTS_idx"].append(tts_idx - 1)
            data_dict["canonical_TIS_coord"].append(tis)
            data_dict["canonical_TTS_coord"].append(tts)
            data_dict["canonical_LTS_coord"].append(lts)
            data_dict["canonical_protein_seq"].append(prot)
        else:
            data_dict["has_annotated_stop_codon"].append(False)
            data_dict["has_annotated_start_codon"].append(False)
            data_dict["CDS_idxs"].append(np.empty(0, dtype=int))
            data_dict["CDS_coords"].append(np.empty(0, dtype=int))
            data_dict["canonical_TIS_exon"].append(-1)
            data_dict["canonical_TIS_idx"].append(-1)
            data_dict["canonical_TTS_idx"].append(-1)
            data_dict["canonical_LTS_idx"].append(-1)
            data_dict["canonical_TIS_coord"].append(-1)
            data_dict["canonical_TTS_coord"].append(-1)
            data_dict["canonical_LTS_coord"].append(-1)
            data_dict["canonical_protein_seq"].append(None)
        data_dict["exon_idxs"].append(ftr_idxs["exon"])
        check_genomic_order(exon_coords, "+" if is_pos_strand else "-")
        data_dict["exon_coords"].append(np.array(exon_coords))
        data_dict["seq"].append(seq)
        data_dict["tis"].append(target_seq)
        data_dict["transcript_id"].append(gtf_tr["transcript_id"].unique()[0])

    if len(altered_tr_exons) > 0:
        print(
            f"WARNING: Exon numbering for {len(altered_tr_exons)} transcripts was altered. "
            "Please check the GTF file for correct exon numbering."
        )
    db_ext = pl.from_dict(data_dict)
    db = db_ext.join(db, on="transcript_id", how="left")
    # drop exon info that is not correct at transcript-level
    db = db.drop(DROPPED_HEADERS, strict=False)
    # vectorize protein sequences (less storage)
    db = db.with_columns(
        pl.col("canonical_protein_seq")
        .fill_null("")
        .map_elements(
            prot2vec,
            pl.List(pl.Int64),
        )
        .cast(pl.List(pl.Int8))
    )

    return db


def store_sparse_chunked(f, data, format="csr", overwrite=False, append=False):
    """
    Append a list of matrices to an HDF5 group incrementally.
    """
    format_attr_dict = {
        "csr": ["data", "indices", "indptr", "shape"],
    }
    format_dict = {"csr": sparse.csr_matrix}
    
    if type(data) not in [list, np.ndarray]:
        data = [data]
        
    if not data:
        return
        
    transform = type(data[0]) != format_dict[format]
    data_attr = {key: [] for key in format_attr_dict[format]}
    
    for sample in data:
        if transform:
            sample = format_dict[format](sample)
        for attribute in data_attr.keys():
            data_attr[attribute].append(np.array(getattr(sample, attribute)))
            
    for attribute in data_attr.keys():
        if overwrite and attribute in f.keys() and not append:
            del f[attribute]
            
        att_dtype = data_attr[attribute][0].dtype
        att_lens = np.array([len(d) for d in data_attr[attribute]])
        
        # If append is True, resizing existing dataset
        if append and attribute in f.keys():
            ds = f[attribute]
            curr_size = ds.shape[0]
            add_size = len(data_attr[attribute])
            ds.resize((curr_size + add_size,) + ds.shape[1:])
            if (att_lens[0] == att_lens).all():
                ds[curr_size:] = data_attr[attribute]
            else:
                for i, arr in enumerate(data_attr[attribute]):
                    ds[curr_size + i] = arr
        else:
            # Create new resizable dataset
            if (att_lens[0] == att_lens).all():
                data_arr = np.array(data_attr[attribute])
                max_shape = (None,) + data_arr.shape[1:]
                f.create_dataset(attribute, data=data_arr, maxshape=max_shape)
            else:
                dt = h5py.vlen_dtype(att_dtype)
                ds = f.create_dataset(attribute, shape=(len(data_attr[attribute]),), maxshape=(None,), dtype=dt)
                for i, arr in enumerate(data_attr[attribute]):
                    ds[i] = arr

def parse_ribo_reads(path, read_lims, f_ids, f_lens, exp_grp, low_memory=False):
    print(f"\t -- Reading and processing file: {path}...")
    import pysam
    _, file_ext = os.path.splitext(path)
    
    bam_path = path
    temp_bam = False
    
    if file_ext == ".sam":
        bam_path = path.replace(".sam", ".bam")
        if not os.path.exists(bam_path):
            print(f"\t -- Converting and sorting SAM to BAM ({bam_path})...")
            pysam.sort("-o", bam_path, path)
            temp_bam = True
            
    bai_path = bam_path + ".bai"
    if not os.path.exists(bai_path):
        try:
            print(f"\t -- Indexing BAM... ({bai_path})")
            pysam.index(bam_path)
        except pysam.SamtoolsError:
            sorted_bam = bam_path.replace(".bam", ".sorted.bam")
            print(f"\t -- BAM not sorted. Sorting to {sorted_bam}...")
            pysam.sort("-o", sorted_bam, bam_path)
            pysam.index(sorted_bam)
            if temp_bam:
                os.remove(bam_path)
            bam_path = sorted_bam
            temp_bam = True

    bam = pysam.AlignmentFile(bam_path, "rb")
    num_read_lens = read_lims[1] - read_lims[0]
    read_len_dict = {
        read_len: i for i, read_len in enumerate(range(read_lims[0], read_lims[1]))
    }
    
    f_ids_list = f_ids.to_list()
    f_lens_list = f_lens.to_list()
    
    CHUNK_SIZE = 50000
    total_transcripts = len(f_ids_list)
    num_reads = []
    
    pbar = tqdm(total=total_transcripts, desc="Processing Transcripts")
    for chunk_start in range(0, total_transcripts, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_transcripts)
        chunk_ids = f_ids_list[chunk_start:chunk_end]
        chunk_lens = f_lens_list[chunk_start:chunk_end]
        
        chunk_matrices = []
        for tr_id, tr_len in zip(chunk_ids, chunk_lens):
            mtx_shape = (num_read_lens, tr_len)
            rows = []
            cols = []
            
            try:
                for read in bam.fetch(tr_id):
                    read_len = read.query_length
                    if read_lims[0] <= read_len < read_lims[1]:
                        try:
                            rows.append(read_len_dict[read_len])
                            # read.reference_start is 0-based
                            cols.append(read.reference_start)
                        except KeyError:
                            pass
            except ValueError:
                # Missing from index
                pass
            
            if len(rows) > 0:
                data = np.ones(len(rows), dtype=np.int32)
                sp_mtx = sparse.coo_matrix((data, (rows, cols)), shape=mtx_shape, dtype=np.int32)
                sp_mtx.sum_duplicates() 
                csr = sp_mtx.tocsr()
                chunk_matrices.append(csr)
                num_reads.append(csr.sum())
            else:
                chunk_matrices.append(sparse.csr_matrix(mtx_shape, dtype=np.int32))
                num_reads.append(0)
                
        append = chunk_start > 0
        store_sparse_chunked(exp_grp, chunk_matrices, overwrite=True, append=append)
        pbar.update(chunk_end - chunk_start)
        
    pbar.close()
    bam.close()
    return num_reads
