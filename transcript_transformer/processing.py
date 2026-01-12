import numpy as np
from tqdm import tqdm
import h5py
import polars as pl
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from transcript_transformer import (
    RIBOTIE_MQC_HEADER,
    START_CODON_MQC_HEADER,
    BIOTYPE_VARIANT_MQC_HEADER,
    ORF_TYPE_MQC_HEADER,
    ORF_TYPE_ORDER,
    STOP_CDNS,
    CDN_PROT_DICT,
    IDX_PROT_DICT,
    IDX_DNA_DICT,
    STANDARD_HEADERS,
    RENAME_HEADERS,
    STANDARD_OUT_HEADERS,
    RIBO_OUT_HEADERS,
    MUT_OUT_DICT,
)
from .util_functions import (
    apply_mutations_to_seq,
    alter_ribo_counts,
    check_genomic_order,
    construct_prot,
    derive_polars_type,
    find_distant_exon_coord,
    get_str2str_idx_map,
    prtime,
    transcript_region_to_exons,
)
from typing import Callable


# from https://github.com/pola-rs/polars/issues/7210
def list_eval_ref(
    list_col, ref_col, op: Callable[[pl.Expr, pl.Expr], pl.Expr]
) -> pl.Expr:
    return pl.concat_list(pl.struct(list_col, ref_col)).list.eval(
        op(
            pl.element().struct.field(list_col).explode(),
            pl.element().struct.field(ref_col),
        )
    )


def _exprs_for_orf_features(suffix=""):
    seq = pl.col(f"seq{suffix}")
    # 1. Define expressions for translation and stop codon detection.
    orf_sequence = seq.str.slice(pl.col(f"TIS_idx{suffix}"))
    transcript_len = seq.str.len_chars()
    codons = orf_sequence.str.extract_all(r".{3}")
    stop_indices_list = codons.list.eval(pl.element().is_in(STOP_CDNS).arg_true())
    stop_codon_idx = stop_indices_list.list.first()
    # 2. Derive the final protein sequence and related features.
    TTS_on_transcript = stop_codon_idx.is_not_null()
    codons_before_stop = (
        pl.when(TTS_on_transcript)
        .then(codons.list.slice(0, stop_codon_idx))
        .otherwise(codons)
    )
    protein_seq = codons_before_stop.list.eval(
        pl.element().replace_strict(CDN_PROT_DICT, default=pl.lit("X"))
    ).list.join("")
    stop_codon = codons.list.get(stop_codon_idx)
    ORF_len = (protein_seq.str.len_chars().cast(pl.Int32)) * 3
    TTS_idx = (
        pl.when(TTS_on_transcript)
        .then(pl.col(f"TIS_idx{suffix}") + ORF_len)
        .otherwise(pl.lit(-1))
    )
    LTS_idx = (
        pl.when(TTS_on_transcript)
        .then(pl.col(f"TIS_idx{suffix}") + ORF_len - 1)
        .otherwise(transcript_len - 1)
    )
    return [
        transcript_len.alias(f"transcript_len{suffix}"),
        TTS_on_transcript.alias(f"TTS_on_transcript{suffix}"),
        protein_seq.alias(f"protein_seq{suffix}"),
        stop_codon.alias(f"stop_codon{suffix}"),
        (pl.col(f"TIS_idx{suffix}") + 1).alias(f"TIS_pos{suffix}"),
        seq.str.slice(pl.col(f"TIS_idx{suffix}"), 3).alias(f"start_codon{suffix}"),
        ORF_len.alias(f"ORF_len{suffix}"),
        (
            pl.col("transcript_id")
            + "_"
            + (pl.col(f"TIS_idx{suffix}") + 1).cast(pl.String)
        ).alias(f"ORF_id{suffix}"),
        TTS_idx.alias(f"TTS_idx{suffix}"),
        LTS_idx.alias(f"LTS_idx{suffix}"),
        (TTS_idx + 1).alias(f"TTS_pos{suffix}"),
    ]


def _get_calculate_pois_exprs(suffix=""):
    exprs = []
    for poi in ["TIS", "LTS", "TTS"]:
        # In case of mutations, map idxs back to original transcript idx to calculate coords
        # Otherwise, it's possible to create OOB errors etc.
        ref_col = f"{poi}_idx{suffix}"
        list_f = lambda element, ref: element <= ref
        poi_exon_calc = list_eval_ref("exon_idxs", ref_col, list_f).list.sum() // 2 + 1
        # If the list is empty, return None (null) to prevent out-of-bounds errors.
        poi_exon = (
            pl.when(pl.col("exon_idxs").list.len() > 0)
            .then(poi_exon_calc)
            .otherwise(pl.lit(None))
        )
        # Define an intermediate expression for the index relative to its exon's start.
        exon_start_idx = (poi_exon - 1) * 2
        poi_idx_on_exon = pl.col(ref_col) - pl.col("exon_idxs").list.get(exon_start_idx)
        # Define an intermediate expression for the genomic coordinate.
        poi_coord = (
            pl.when(pl.col("strand") == "+")
            .then(
                pl.col("exon_coords").list.get(exon_start_idx, null_on_oob=True)
                + poi_idx_on_exon
            )
            .otherwise(  # Assumes strand is "-"
                pl.col("exon_coords").list.get(exon_start_idx + 1, null_on_oob=True)
                - poi_idx_on_exon
            )
        )
        is_inv = pl.col(ref_col) == -1
        # --- 2. Add final, aliased expressions to the list ---
        exprs += [
            pl.when(is_inv)
            .then(pl.lit(-1))
            .otherwise(poi_exon)
            .alias(f"{poi}_exon{suffix}"),
            pl.when(is_inv)
            .then(pl.lit(-1))
            .otherwise(poi_idx_on_exon)
            .alias(f"{poi}_idx_on_exon{suffix}"),
            pl.when(is_inv)
            .then(pl.lit(-1))
            .otherwise(poi_idx_on_exon + 1)
            .alias(f"{poi}_pos_on_exon{suffix}"),
            pl.when(is_inv)
            .then(pl.lit(-1))
            .otherwise(poi_coord)
            .alias(f"{poi}_coord{suffix}"),
        ]
        if poi == "TIS":
            exprs += [
                pl.when(pl.col("canonical_TIS_idx") != 0)
                .then(pl.col(f"TIS_idx{suffix}") - pl.col("canonical_TIS_idx"))
                .otherwise(pl.lit(None))
                .cast(pl.Int32)
                .alias(f"dist_from_canonical_TIS{suffix}"),
            ]

    return exprs


def _calculate_orf_biotype(df_columns, suffix=""):
    # detect ORF biotypes, evaluate whether transcript biotype is given
    if "transcript_biotype" in df_columns:
        biotype_expr = pl.col("transcript_biotype") == "lncRNA"
    else:
        biotype_expr = pl.lit(False)
    return (
        pl.when(pl.col("canonical_TIS_idx") != -1)
        .then(
            pl.when(pl.col("canonical_TIS_idx") == pl.col(f"TIS_idx{suffix}"))
            .then(
                pl.when(pl.col("canonical_LTS_idx") == pl.col(f"LTS_idx{suffix}"))
                .then(pl.lit("annotated CDS"))
                .when(pl.col("canonical_TTS_idx") < pl.col(f"TTS_idx{suffix}"))
                .then(pl.lit("C-terminal extension"))
                .otherwise(pl.lit("C-terminal truncation"))
            )
            .when(pl.col("canonical_TTS_idx") < pl.col(f"TIS_idx{suffix}"))
            .then(pl.lit("dORF"))
            .when(pl.col("canonical_TIS_idx") >= pl.col(f"TTS_idx{suffix}"))
            .then(pl.lit("uORF"))
            .when(pl.col("canonical_TIS_idx") > pl.col(f"TIS_idx{suffix}"))
            .then(
                pl.when(pl.col("canonical_TTS_idx") == pl.col(f"TTS_idx{suffix}"))
                .then(pl.lit("N-terminal extension"))
                .otherwise(pl.lit("uoORF"))
            )
            .when(pl.col("canonical_TTS_idx") < pl.col(f"TTS_idx{suffix}"))
            .then(pl.lit("doORF"))
            .otherwise(
                pl.when(pl.col("canonical_TTS_idx") == pl.col(f"TTS_idx{suffix}"))
                .then(pl.lit("N-terminal truncation"))
                .otherwise(pl.lit("intORF"))
            )
        )
        .otherwise(
            pl.when(biotype_expr)
            .then(pl.lit("lncRNA-ORF"))
            .otherwise(pl.lit("varRNA-ORF"))
        )
        .alias(f"ORF_type{suffix}")
    )


def parse_ribo_data(df, f, h5_path, ribo_ids, parallel):
    sel_cols = ["ORF_id", "h5_idx", "TIS_idx", "TTS_idx", "ORF_len"]
    if "mutations" in df.columns:
        sel_cols += ["mutations"]
    df_ribo = df.select(sel_cols)
    # multiple sets in case of merged data sets
    sys_path = f"{h5_path.split('.h5')[0]}_{{sample}}.h5"
    db_path = "transcript/riboseq/{sample}/5/"
    ribo_paths = [
        [sys_path.format(sample=sample), db_path.format(sample=sample)]
        for sample in ribo_ids
    ]
    # only data and indices of sparse object are required (all counts are summed over read lengths)
    csr_cols = ["data", "indices", "indptr", "shape"]
    for h in csr_cols:
        if parallel:
            count_matrix = [
                np.array(h5py.File(a)[f"{p}/{h}"])[df_ribo["h5_idx"]]
                for a, p in ribo_paths
            ]
        else:
            count_matrix = [
                np.array(f[f"{p}/{h}"])[df_ribo["h5_idx"]] for _, p in ribo_paths
            ]
        if h in ["data", "indices"]:
            # concatenate the individual entries by keeping transcript position and value
            exp_iter = range(len(count_matrix))
            tr_iter = range(len(count_matrix[0]))
            counts = [
                np.concatenate([count_matrix[i][j] for i in exp_iter]) for j in tr_iter
            ]
            total_reads = [len(c) for c in counts]
        elif h == "indptr":
            # row values are not important (read length info is discarded), replace last value
            # of indptr with total reads (normally indicating all reads are in the last row)
            counts = count_matrix[0]
            counts[:, -1] = total_reads
        else:
            # shape is the same for all samples, just take the first one
            counts = count_matrix[0]
        df_ribo = df_ribo.with_columns(
            pl.Series(name=h, values=list(counts), dtype=pl.List(pl.Int32))
        )
    # Filter out results with 0 reads (only happens when training on zero-read data)
    df_ribo = df_ribo.filter(pl.col("data").cast(pl.List(pl.Int32)).list.len() > 0)
    # Polars autoconverts empty list columns to array types...
    df_ribo = df_ribo.cast({"data": pl.List(pl.Int32), "indices": pl.List(pl.Int32)})
    # get in-ORF reads and properties that cannot be retrieved using polars API
    csr_f = (
        lambda x: csr_matrix((x["data"], x["indices"], x["indptr"]), shape=x["shape"])
        .sum(axis=0)
        .tolist()[0]
    )
    df_ribo = df_ribo.with_columns(
        pl.struct(*csr_cols)
        .map_elements(
            csr_f,
            return_dtype=pl.List(pl.Int32),
        )
        .alias("reads")
    )

    if "mutations" in df_ribo.columns:
        muts = pl.col("mutations").str.split(";")
        df_ribo = df_ribo.with_columns(muts.alias("mutations_split")).with_columns(
            pl.struct(["reads", "mutations_split"])
            .map_elements(
                lambda s: alter_ribo_counts(s["reads"], s["mutations_split"]),
                return_dtype=pl.List(pl.List(pl.Int32)),
            )
            .list.to_struct(fields=["reads", "tmp"])
            .struct.unnest()
        )

    # get ribo properties supported by polars API
    df_ribo = df_ribo.with_columns(
        reads_in_ORF=(pl.col("reads").list.slice(pl.col("TIS_idx"), pl.col("ORF_len"))),
        reads_in_transcript=pl.col("data").list.sum(),
    ).with_columns(
        reads_in_ORF=pl.col("reads_in_ORF").list.sum(),
        reads_in_frame_frac=(
            pl.col("reads_in_ORF")
            .list.gather_every(3)
            .list.sum()
            .truediv(pl.col("reads_in_ORF").list.sum())
        ),
        reads_5UTR=(pl.col("reads").list.slice(0, pl.col("TIS_idx")).list.sum()),
        reads_3UTR=(
            pl.when(pl.col("TTS_idx") != -1)
            .then(pl.col("reads").list.slice(pl.col("TTS_idx")).list.sum())
            .otherwise(pl.lit(0))
        ),
        reads_skew=(
            pl.col("reads_in_ORF")
            .list.slice(offset=pl.col("reads_in_ORF").list.len().truediv(2))
            .list.sum()
            .truediv(pl.col("reads_in_ORF").list.sum())
            .sub(0.5)
            .mul(2)
        ),
        reads_coverage_frac=(
            pl.col("reads_in_ORF")
            .list.eval((pl.element() > 0))
            .list.sum()
            .truediv(pl.col("reads_in_ORF").list.len())
        ),
        reads_entropy=(
            pl.col("reads_in_ORF").map_elements(
                lambda x: entropy(x, np.full(len(x), 1) / len(x)),
                return_dtype=pl.Float32,
            )
        ),
    )
    return df_ribo.fill_nan(0)


def construct_output_table(
    h5_path,
    out_prefix,
    mut_dict={},
    output=None,
    is_rt_output=False,
    prob_cutoff=0.15,
    correction=False,
    dist=9,
    start_codons=".*TG$",
    min_ORF_len=15,
    remove_duplicates=True,
    exclude_invalid_TTS=True,
    grouped_ribo_ids={},
    parallel=False,
    return_ORF_coords=False,
    max_preds=100000,
):
    """
    Constructs the output table using Polars LazyFrames for optimized performance.
    """
    tool_scores = []
    f = h5py.File(h5_path, "r")
    f_tr_ids = np.array(f["transcript/transcript_id"])
    f_headers = pl.Series(f["transcript"].keys())
    f_headers = f_headers.filter(~f_headers.is_in(["riboseq", "tis"]))

    has_stored_tt_output = "tis_transformer_score" in f_headers
    f_headers = f_headers.filter(f_headers != "tis_transformer_score")

    if (output is not None) and is_rt_output:
        tool_scores.insert(0, "ribotie_score")
        prefix = "ribotie_"
    else:
        tool_scores.insert(0, "tis_transformer_score")
        prefix = "tis_transformer_"
        if output is None:
            assert has_stored_tt_output, "No model predictions found"

    xtr_heads = [h for h in f_headers if h not in STANDARD_HEADERS]

    if output is not None:
        tr_ids = np.array([o[0].split(b"|")[1] for o in output])
        preds = [o[1] for o in output]
        group = output[0][0].split(b"|")[0].decode()
        pred_to_h5_args = get_str2str_idx_map(tr_ids, f_tr_ids)

        lf_initial = pl.LazyFrame(
            {
                "transcript_id": tr_ids,
                f"{prefix}score": preds,
                "h5_idx": pred_to_h5_args,
            }
        ).with_columns(
            pl.col("transcript_id").cast(pl.String),
            pl.col(f"{prefix}score").map_elements(
                list, return_dtype=pl.List(pl.Float32)
            ),
        )

        if is_rt_output and has_stored_tt_output and not mut_dict:
            tool_scores.append("tis_transformer_score")
            tt_scores = f["transcript/tis_transformer_score"][:][pred_to_h5_args]
            lf_initial = lf_initial.with_columns(
                tis_transformer_score=pl.lit(tt_scores).map_elements(
                    list, return_dtype=pl.List(pl.Float32)
                )
            )
        tool_headers = tool_scores + [f"{prefix}rank"]
        ribo_out_headers = [] if not is_rt_output else RIBO_OUT_HEADERS
        out_headers = tool_headers + STANDARD_OUT_HEADERS + ribo_out_headers + xtr_heads
    else:
        tool_headers = ["tis_transformer_score", "tis_transformer_rank"]
        lf_initial = pl.LazyFrame(
            {
                "transcript_id": f["transcript/transcript_id"][:],
                "h5_idx": np.arange(len(f_tr_ids)),
                f"{prefix}score": f["transcript/tis_transformer_score"][:],
            }
        ).with_columns(
            pl.col(f"{prefix}score").map_elements(
                list, return_dtype=pl.List(pl.Float32)
            )
        )
        out_headers = tool_headers + STANDARD_OUT_HEADERS + xtr_heads

    if return_ORF_coords:
        out_headers += ["ORF_coords"]
    if mut_dict:
        out_headers += ["mutations", "TIS_pos_refmap", "TTS_pos_refmap"]
        # change existing header names in output
        out_headers = [MUT_OUT_DICT.get(h, h) for h in out_headers]

    el_gt_th = pl.element() > prob_cutoff
    el_not_nan = pl.element().is_not_nan()
    pos_args = pl.col(f"{prefix}score").list.eval((el_gt_th & el_not_nan).arg_true())

    lf_filtered = (
        lf_initial.sort("h5_idx")
        .with_columns(TIS_idx=pos_args.cast(pl.List(pl.Int64)))
        .filter(pl.col("TIS_idx").list.len() > 0)
        .with_columns(
            **{
                score: pl.col(score).list.gather(pl.col("TIS_idx"))
                for score in tool_scores
            },
        )
    )
    # Top K prediction filtering (when necessary)
    # NOTE: This part requires a collection to determine the total number of predictions.
    total_preds = (
        lf_filtered.select(pl.col("TIS_idx").list.len().sum()).collect().item()
    )
    if total_preds > max_preds:
        prtime(f"Too many predictions, filtering to top {max_preds}...", "\t")
        # Efficiently find the score of the Nth-best prediction.
        score_th = (
            lf_filtered.select(pl.col(f"{prefix}score").explode())
            .sort(f"{prefix}score", descending=True)
            .limit(max_preds)
            .select(pl.all().last())  # Get the Nth value
            .collect()
            .item()
        )
        pos_idxs = pl.col(f"{prefix}score").list.eval(
            (pl.element() >= score_th).arg_true()
        )
        lf_filtered = lf_filtered.with_columns(
            TIS_idx=pl.col("TIS_idx").list.gather(pos_idxs),
            **{score: pl.col(score).list.gather(pos_idxs) for score in tool_scores},
        )

    # # Metadata Join
    meta_exprs = []
    for h in f_headers:
        # Keep the array in memory for the lazy execution later
        v = np.array(f[f"transcript/{h}"][()])
        meta_exprs.append(
            pl.col("h5_idx")
            # The map_elements function is correct, it just needs to be executed lazily.
            .map_elements(
                lambda x, arr=v: arr[x], return_dtype=derive_polars_type(v)
            ).alias(h)
        )
    lf_filtered = lf_filtered.with_columns(meta_exprs).with_columns(
        pl.col(pl.Binary).cast(pl.String),
    )

    # Devectorize and process sequences
    lf = lf_filtered.with_columns(
        (
            pl.col("canonical_protein_seq")
            .map_elements(list, return_dtype=pl.List(pl.Int8))
            .list.eval(pl.element().replace_strict(IDX_PROT_DICT))
            .list.join("")
        ),
    )
    if mut_dict:
        lf = (
            lf.with_columns(
                mutations=pl.col("transcript_id").replace_strict(mut_dict, default=[])
            )
            .with_columns(
                pl.struct(["seq", "mutations"])
                .map_elements(
                    lambda s: apply_mutations_to_seq(s["seq"], s["mutations"]),
                    return_dtype=pl.List(pl.List(pl.Int64)),
                )
                .list.to_struct(fields=["seq", "mut_seq_map", "mutations_mask"])
                .struct.unnest()
            )
            .with_columns(
                mutations=pl.col("mutations")
                .list.gather(
                    pl.col("mutations_mask")
                    .cast(pl.List(pl.Boolean))
                    .list.eval(pl.element().arg_true())
                )
                .list.join(";"),
            )
        )
    lf = lf.with_columns(
        seq=pl.col("seq")
        .list.eval(pl.element().replace_strict(IDX_DNA_DICT))
        .list.join("")
    )
    # Explode lf to get ORF predictions per row
    lf = lf.explode(tool_scores + ["TIS_idx"]).sort("h5_idx")
    if correction:
        prtime("Correcting near-miss TIS predictions...", "\t")
        # max distance of correction
        corr_dist = pl.lit(dist * 3, dtype=pl.Int64)
        # clip upstream correction between 0 and pos of TIS prediction
        upstr_corr = pl.min_horizontal(
            pl.col("TIS_idx") - pl.col("TIS_idx").mod(3), corr_dist
        )

        search_window = pl.col("seq").str.slice(
            pl.col("TIS_idx") - upstr_corr,
            upstr_corr + 3 + corr_dist,
        )
        codons = search_window.str.extract_all(r".{3}")
        atg_codon_indices = codons.list.eval(
            pl.element().str.contains("ATG").arg_true()
        )
        corr_distances = (atg_codon_indices * 3) - upstr_corr
        corr_distances_filled = (
            pl.when(corr_distances.list.len() == 0)
            .then(pl.lit([0], dtype=pl.List(pl.Int64)))
            .otherwise(corr_distances)
        )
        closest_distance_idx = corr_distances_filled.list.eval(
            pl.element().abs()
        ).list.arg_min()
        best_correction = corr_distances_filled.list.get(closest_distance_idx)
        lf = lf.with_columns(
            TIS_idx=pl.col("TIS_idx") + best_correction,
            correction=corr_distances_filled.list.get(closest_distance_idx),
        )
        if remove_duplicates:
            lf = lf.sort("ribotie_score", descending=True).unique(
                ["transcript_id", "TIS_idx"], keep="first"
            )

    # --- Feature calculation using helper functions ---
    lf = lf.collect().lazy()  # fixes schema confusion for 'seq'
    lf = lf.with_columns(_exprs_for_orf_features()).filter(pl.col("ORF_len") > 0)

    # Add distance from canonical TIS
    dist_from_canonical_TIS = (
        pl.when(pl.col("canonical_TIS_idx") != 0)
        .then(pl.col("TIS_idx") - pl.col("canonical_TIS_idx"))
        .otherwise(pl.lit(None))
        .cast(pl.Int32)
    )
    lf = lf.with_columns(
        dist_from_canonical_TIS=dist_from_canonical_TIS,
        frame_wrt_canonical_TIS=dist_from_canonical_TIS % 3,
        canonical_TIS_pos=pl.col("canonical_TIS_idx") + 1,
        canonical_TTS_pos=pl.col("canonical_TTS_idx") + 1,
        canonical_LTS_pos=pl.col("canonical_LTS_idx") + 1,
    )

    if mut_dict:
        lf = lf.with_columns(
            TIS_idx_refmap=pl.col("mut_seq_map").list.get(pl.col("TIS_idx")),
            LTS_idx_refmap=pl.col("mut_seq_map").list.get(pl.col("LTS_idx")),
            TTS_idx_refmap=pl.col("mut_seq_map").list.get(pl.col("TTS_idx")),
            TIS_pos_refmap=pl.col("mut_seq_map").list.get(pl.col("TIS_idx")) + 1,
            TTS_pos_refmap=pl.col("mut_seq_map").list.get(pl.col("TTS_idx")) + 1,
        )
        # Calculate coordinates for TIS, LTS, TTS
        lf = lf.with_columns(_get_calculate_pois_exprs("_refmap"))
    else:
        lf = lf.with_columns(_get_calculate_pois_exprs())

    # --- Ribo-seq Join ---
    df_collected = lf.collect()

    if is_rt_output:
        prtime("Parsing ribo-seq information...", "\t")
        df_ribo = parse_ribo_data(
            df_collected, f, h5_path, grouped_ribo_ids.get(group), parallel
        )
        if len(df_ribo) > 0 and df_ribo.shape[1] > 10:
            df_collected = df_collected.join(
                df_ribo.select(["ORF_id", *RIBO_OUT_HEADERS[1:]]),
                on="ORF_id",
                how="inner",
            )
        else:
            if len(df_collected) > 0:
                print("!-> No ribosome reads present amongst input samples.")
            df_collected = df_collected.join(
                df_ribo.select("ORF_id"), on="ORF_id", how="inner"
            )

    # Stop early if df is empty
    if len(df_collected) == 0:
        # Handle empty case as before
        out_dicts = {n: pl.Series(n, [], dtype=pl.Utf8) for n in out_headers}
        df_out = pl.DataFrame(out_dicts).rename(RENAME_HEADERS)
        for label in [".redundant", "", ".novel"]:
            df_out.write_csv(f"{out_prefix}{label}.csv")
        print("\t !-> The positive set is empty!")
        f.close()
        return df_out, df_out, df_out

    lf = df_collected.lazy()  # Convert back to LazyFrame

    # --- ORF Biotype and CDS Variant Detection ---
    prtime("Parsing ORF type information...", "\t")
    lf = lf.with_columns(_calculate_orf_biotype(lf.collect_schema().names()))

    prtime("Detecting CDS variants...", "\t")
    if mut_dict:
        suffix = "_refmap"
    else:
        suffix = ""
    out_cols = [f"ORF_coords{suffix}", f"ORF_exons{suffix}"]
    out_types = [pl.List(pl.Int64), pl.List(pl.Int64)]
    attrs = [f"TIS_coord{suffix}", f"LTS_coord{suffix}", "strand", "exon_coords"]

    lf = (
        lf.with_columns(
            pl.struct(set(attrs))
            .map_elements(
                lambda x: dict(
                    zip(out_cols, transcript_region_to_exons(*[x[a] for a in attrs]))
                ),
                return_dtype=pl.Struct(dict(zip(out_cols, out_types))),
            )
            .struct.unnest()
        )
        .with_columns(
            pl.col(f"ORF_coords{suffix}")
            .list.gather_every(2, 0)
            .alias(f"ORF_exon_start{suffix}"),
            pl.col(f"ORF_coords{suffix}")
            .list.gather_every(2, 1)
            .alias(f"ORF_exon_end{suffix}"),
        )
        .with_columns(
            (pl.col(f"ORF_exon_start{suffix}") - pl.col(f"ORF_exon_end{suffix}"))
            .list.eval(pl.element().abs() + 1)
            .alias(f"ORF_exon_len{suffix}")
        )
    )
    h5_cols = [
        "transcript_id",
        "seqname",
        "strand",
        "CDS_coords",
        "canonical_TIS_coord",
        "canonical_LTS_coord",
    ]
    mask = pl.Series(list(f[f"transcript/canonical_TIS_idx"])) != -1
    df_CDS = (
        pl.DataFrame(
            {h: np.array(f[f"transcript/{h}"])[mask.arg_true()] for h in h5_cols}
        )
        .with_columns(
            pl.col("CDS_coords").map_elements(list, return_dtype=pl.List(pl.Int64)),
            pl.col(pl.Binary).cast(pl.String),
        )
        .with_columns(
            CDS_exon_start=pl.col("CDS_coords").list.gather_every(2, 0),
            CDS_exon_end=pl.col("CDS_coords").list.gather_every(2, 1),
            CDS_start_range=pl.when(pl.col("strand") == "+")
            .then(pl.col("CDS_coords").list.get(0))
            .otherwise(pl.col("CDS_coords").list.get(-2)),
            CDS_end_range=pl.when(pl.col("strand") == "+")
            .then(pl.col("CDS_coords").list.get(-1))
            .otherwise(pl.col("CDS_coords").list.get(1)),
        )
        .drop("CDS_coords")
    )
    f.close()

    # NOTE: Group-by operation is collected to process chromosome by chromosome, avoiding OOM issues.
    df = lf.collect()
    df_grps = []
    total = df["seqname"].n_unique()
    for seqname, df_grp in tqdm(df.group_by("seqname"), total=total, desc="seqname"):
        df_CDS_grp = df_CDS.filter(pl.col("seqname") == seqname[0])
        df_grp = parse_CDS_overlap(df_grp, df_CDS_grp, suffix=suffix)
        df_grps.append(df_grp)

    if not df_grps:
        df = df.clear(0)  # Create empty dataframe with same schema
    else:
        df = pl.concat(df_grps)

    if return_ORF_coords:
        df = df.with_columns(
            ORF_coords=pl.col(f"ORF_coords{suffix}")
            .cast(pl.List(pl.String))
            .list.join(";")
        )

    # --- Final Filtering and Saving ---
    conds_xtr = [
        pl.col("TTS_on_transcript") if exclude_invalid_TTS else pl.lit(True),
        pl.col("start_codon").str.contains(start_codons),
        pl.col("ORF_len") >= min_ORF_len,
    ]
    
    # All results get manual filter steps
    df = df.filter(pl.lit(True).and_(*conds_xtr))

    if len(df) > 0:
        df_filt = filter_CDS_variants(df, suffix=suffix)
    else:
        df_filt = df

    df_novel = df_filt.filter(pl.col("ORF_type") != "annotated CDS")
    # Save outputs
    for df_, label in zip([df, df_filt, df_novel], [".redundant", "", ".novel"]):
        # The final DataFrames are collected implicitly by being eager at this point
        save_output_table(df_, out_prefix, label, prefix, out_headers)

    return df, df_filt, df_novel


def save_output_table(df, out_prefix, label, prefix, out_headers):
    df = (
        df.with_columns(
            (pl.col(f"{prefix}score").rank(method="ordinal", descending=True)).alias(
                f"{prefix}rank"
            )
        )
        .select(out_headers)
        .sort(f"{prefix}rank")
        .rename(RENAME_HEADERS)
    )
    df.write_csv(f"{out_prefix}{label}.csv", float_precision=4)


def parse_CDS_overlap(
    df: pl.DataFrame, df_CDS: pl.DataFrame, suffix: str = ""
) -> pl.DataFrame:
    """
    Identifies and annotates ORFs that overlap with known CDS regions using a
    fully native, memory-efficient, and logically correct lazy Polars pipeline.
    """
    TIS_coord = pl.col(f"TIS_coord{suffix}")
    LTS_coord = pl.col(f"LTS_coord{suffix}")
    ORF_exon_end = pl.col(f"ORF_exon_end{suffix}")
    ORF_exon_start = pl.col(f"ORF_exon_start{suffix}")

    # This initial step is already vectorized and efficient.
    df = df.with_columns(
        has_CDS_TIS=(TIS_coord.is_in(df_CDS["canonical_TIS_coord"])),
        has_CDS_TTS=(LTS_coord.is_in(df_CDS["canonical_LTS_coord"])),
    )

    # --- Step 1: Find All Overlapping ORF-CDS Pairs Lazily ---
    overlap_filter = (
        pl.when(pl.col("strand") == "+")
        .then(
            (pl.col("CDS_start_range") < ORF_exon_end.list.last())
            & (pl.col("CDS_end_range") > ORF_exon_start.list.first())
        )
        .otherwise(
            (pl.col("CDS_start_range") < ORF_exon_end.list.first())
            & (pl.col("CDS_end_range") > ORF_exon_start.list.last())
        )
    )

    lf_joined = (
        df.lazy()
        .select(
            "ORF_id",
            "strand",
            f"ORF_exon_start{suffix}",
            f"ORF_exon_end{suffix}",
            "ORF_len",
        )
        .join(df_CDS.lazy(), how="cross")
        .filter(overlap_filter)
    )

    # --- Step 2: Calculate Features for Each Overlapping Exon ---
    lf_with_clone_info = lf_joined.with_columns(
        is_clone=(ORF_exon_start == pl.col("CDS_exon_start"))
        & (ORF_exon_end == pl.col("CDS_exon_end"))
    )
    lf_exons = lf_with_clone_info.explode(["CDS_exon_start", "CDS_exon_end"]).explode(
        [f"ORF_exon_start{suffix}", f"ORF_exon_end{suffix}"]
    )

    # --- Step 3: Natively Calculate Overlap (from your `eval_overlap` function) ---
    overlap_expr = (
        pl.min_horizontal([f"ORF_exon_end{suffix}", "CDS_exon_end"])
        - pl.max_horizontal([f"ORF_exon_start{suffix}", "CDS_exon_start"])
        + 1
    ).clip(lower_bound=0)
    lf_overlap_calc = lf_exons.with_columns(overlap=overlap_expr)

    # THE FIX for dropped rows: Use a more robust sort_by/group_by/first pattern
    # to correctly select the best overlap for each ORF exon without dropping groups.
    lf_best_overlap = (
        lf_overlap_calc.sort(["overlap", "is_clone"], descending=True)
        .group_by("ORF_id", f"ORF_exon_start{suffix}", f"ORF_exon_end{suffix}")
        .first()
    )

    # --- Step 4: Calculate Non-Overlapping Coordinates ---
    cond_in_frame = (
        pl.when(pl.col("strand") == "+")
        .then((ORF_exon_start - pl.col("CDS_exon_start")) % 3 == 0)
        .otherwise((ORF_exon_end - pl.col("CDS_exon_end")) % 3 == 0)
    )
    lf_no_cds_coords = lf_best_overlap.with_columns(
        is_shared_and_in_frame=cond_in_frame,
        ORF_exon_len=(ORF_exon_end - ORF_exon_start).abs() + 1,
    ).with_columns(
        ORF_coords_no_CDS=pl.when(
            (pl.col("overlap") > 0) & (pl.col("overlap") < pl.col("ORF_exon_len"))
        )
        .then(
            pl.concat_list(
                [
                    pl.when(ORF_exon_start < pl.col("CDS_exon_start"))
                    .then(ORF_exon_start)
                    .otherwise(pl.col("CDS_exon_end")),
                    pl.when(ORF_exon_start < pl.col("CDS_exon_start"))
                    .then(pl.col("CDS_exon_start"))
                    .otherwise(ORF_exon_end),
                ]
            )
        )
        .otherwise(pl.lit([]))
    )

    # --- Step 5: Final Aggregation to the ORF Level ---
    lf_final_feats = lf_no_cds_coords.group_by("ORF_id").agg(
        has_CDS_clones=pl.col("is_clone").any(),
        shared_in_frame_CDS_region=pl.col("transcript_id")
        .filter(pl.col("is_shared_and_in_frame"))
        .unique(),
        # THE FIX for frac: Normalize the summed overlap by the total ORF_len for the group.
        shared_in_frame_CDS_frac=pl.col("overlap").sum() / pl.col("ORF_len").first(),
        ORF_coords_no_CDS=pl.col("ORF_coords_no_CDS").list.explode().drop_nulls(),
    )

    # --- Step 6: Execute the Lazy Plan and Merge Results ---
    df_var_feats = lf_final_feats.collect()

    df = df.join(df_var_feats, on="ORF_id", how="left").with_columns(
        # The fill_null step correctly provides default values for non-overlapping ORFs.
        pl.col("shared_in_frame_CDS_frac").fill_null(0.0),
        pl.col("has_CDS_clones").fill_null(False),
        pl.col("shared_in_frame_CDS_region").fill_null([]),
        pl.col("ORF_coords_no_CDS").fill_null([]),
    )

    return df


def filter_CDS_variants(df, suffix: str = ""):
    """
    Lazily filters for the most relevant CDS variant for each unique TIS coordinate
    using window functions instead of an eager group-by iteration.

    Args:
        df (pl.LazyFrame): The input LazyFrame of ORF predictions.

    Returns:
        pl.LazyFrame: A new LazyFrame with the filtered results.
    """
    TIS_coord = f"TIS_coord{suffix}"
    # Define the filtering conditions for clarity
    c_is_annotated_cds = pl.col("ORF_type") == "annotated CDS"
    c_is_trunc_or_ext = pl.col("ORF_type").is_in(
        [
            "N-terminal truncation",
            "N-terminal extension",
            "C-terminal truncation",
            "C-terminal extension",
        ]
    )
    c_is_other_orf = pl.col("ORF_type").is_in(
        ["uORF", "uoORF", "dORF", "doORF", "intORF", "lncRNA-ORF"]
    )
    c_is_not_clone = pl.col("has_CDS_clones") == False
    c_is_cds_variant = (pl.col("has_CDS_clones") == False) & (
        pl.col("shared_in_frame_CDS_frac") < 1
    )

    if "transcript_biotype" in df.columns:
        c_is_protein_coding = pl.col("transcript_biotype") == "protein_coding"
    else:
        # If the column doesn't exist, this condition can never be true
        c_is_protein_coding = pl.lit(False)

    # --- Main Filtering Logic using Window Functions ---

    # First-pass filter to select the highest-priority ORF type within each group
    primary_filter = (
        pl.when(c_is_annotated_cds.any().over(TIS_coord))
        .then(c_is_annotated_cds)
        .when((c_is_trunc_or_ext & c_is_not_clone).any().over(TIS_coord))
        .then(c_is_trunc_or_ext & c_is_not_clone)
        .when((c_is_other_orf & c_is_cds_variant).any().over(TIS_coord))
        .then(c_is_other_orf & c_is_cds_variant)
        .otherwise(c_is_cds_variant)
    )

    # Second-pass filter to prioritize protein-coding transcripts if available
    secondary_filter = (
        pl.when((c_is_annotated_cds | c_is_protein_coding).any().over(TIS_coord))
        .then(c_is_annotated_cds | c_is_protein_coding)
        .otherwise(pl.lit(True))
    )

    # Apply both filters to the LazyFrame
    lf_filt = df.filter(primary_filter & secondary_filter)

    return lf_filt


def process_seq_preds(ids, preds, seqs, min_prob):
    # Find indices above min_prob for each prediction
    mask = [np.where(pred > min_prob)[0] for pred in preds]

    rows = []
    for i, idxs in enumerate(mask):
        tr = seqs[i]
        for idx in idxs:
            prot_seq, has_stop, stop_codon = construct_prot(tr[idx:])
            TTS_pos = idx + len(prot_seq) * 3
            rows.append(
                {
                    "transcript_id": ids[i],
                    "transcript_length": len(tr),
                    "TIS_pos": idx + 1,
                    "output": preds[i][idx],
                    "start_codon": tr[idx : idx + 3],
                    "TTS_pos": TTS_pos,
                    "stop_codon": stop_codon,
                    "TTS_on_transcript": has_stop,
                    "protein_length": len(prot_seq),
                    "protein_sequence": prot_seq,
                }
            )
    return pl.DataFrame(rows)


def create_multiqc_reports(df, out_prefix, id, name):
    # Start codons
    output = out_prefix + ".start_codons_mqc.tsv"
    header = RIBOTIE_MQC_HEADER.format(id=id, name=name)
    with open(output, "w") as f:
        f.write(header)
        f.write(START_CODON_MQC_HEADER.format(id=id))
    start_codons = df["start_codon"].value_counts()
    with open(output, mode="a") as f:
        start_codons.write_csv(f, separator="\t", include_header=False)

    # Transcript biotypes
    if "transcript_biotype" in df.columns:
        output = out_prefix + ".biotypes_variant_mqc.tsv"
        with open(output, "w") as f:
            f.write(header)
            f.write(BIOTYPE_VARIANT_MQC_HEADER.format(id=id))
        orf_biotypes = (
            df.filter(pl.col("ORF_type") == "varRNA-ORF")["transcript_biotype"]
            .value_counts()
            .sort("count", descending=True)
        )
        with open(output, mode="a") as f:
            orf_biotypes.write_csv(f, separator="\t", include_header=False)

    # ORF types
    output = out_prefix + ".ORF_types_mqc.tsv"
    with open(output, "w") as f:
        f.write(header)
        f.write(ORF_TYPE_MQC_HEADER.format(id=id))
    orf_types = (
        df["ORF_type"]
        .value_counts()
        .sort(pl.col("ORF_type").cast(pl.Enum(ORF_TYPE_ORDER)))
    )
    with open(output, mode="a") as f:
        orf_types.write_csv(f, separator="\t", include_header=False)

    # ORF lengths
    # output = out_prefix + ".ORF_lens_mqc.tsv"
    # ax = df.ORF_len.apply(lambda x: np.log(x)).plot.kde()
    # x, y = np.exp(ax.lines[-1].get_xdata()), ax.lines[-1].get_ydata()
    # with open(output, "w") as f:
    #     f.write(RIBOTIE_MQC_HEADER)
    #     f.write(ORF_LEN_MQC_HEADER)
    #     for x_, y_ in zip(x, y):
    #         f.write(f"{x_}\t{y_}\n")

    return


def csv_to_gtf(h5_path, df, out_prefix, caller, suffix: str = ""):
    """
    Converts a DataFrame of ORF predictions into a GTF file.
    !!! Creates a unique transcript entry for each ORF to ensure GTF compatibility. !!!
    """
    df = df.fill_null("NA").sort("transcript_id").cast({"seqname": pl.String})

    # Create a lookup map for original transcript_id to exon coordinates for efficiency.
    with h5py.File(h5_path, "r") as f:
        h5_ids = np.array(f["transcript/transcript_id"], dtype=str)
        h5_exons = np.array(f["transcript/exon_coords"])
        tr_id_to_exon_coords = dict(zip(h5_ids, h5_exons))

    gtf_lines = []

    # Iterate through each ORF (row) to create a separate transcript entry.
    for orf in df.iter_rows(named=True):
        original_transcript_id = orf["transcript_id"]
        exon_coord = tr_id_to_exon_coords.get(original_transcript_id)

        if exon_coord is None:
            # Optionally, log or handle cases where exon coordinates are not found.
            continue

        check_genomic_order(exon_coord, orf["strand"])

        # Use ORF_id as the new transcript_id for GTF compatibility.
        dummy_transcript_id = orf["ORF_id"]

        # --- Define base attributes for this new dummy transcript ---
        base_attrs = {
            "gene_id": orf["gene_id"],
            "transcript_id": dummy_transcript_id,
            "gene_name": orf["gene_name"],
        }

        # --- 1. Generate the 'transcript' feature line ---
        if orf["strand"] == "+":
            tr_start, tr_stop = exon_coord[0], exon_coord[-1]
        else:
            tr_start, tr_stop = exon_coord[-2], exon_coord[1]

        attr_str = "; ".join([f'{k} "{v}"' for k, v in base_attrs.items()]) + ";"
        gtf_lines.append(
            f'{orf["seqname"]}\t{caller}\ttranscript\t{tr_start}\t{tr_stop}\t.\t{orf["strand"]}\t.\t{attr_str}\n'
        )

        # --- 2. Generate 'exon' feature lines for the transcript ---
        exons = exon_coord.reshape(-1, 2)
        for i, (exon_start, exon_stop) in enumerate(exons):
            exon_attrs = base_attrs | {"exon_number": i + 1}
            attr_str = "; ".join([f'{k} "{v}"' for k, v in exon_attrs.items()]) + ";"
            gtf_lines.append(
                f'{orf["seqname"]}\t{caller}\texon\t{exon_start}\t{exon_stop}\t.\t{orf["strand"]}\t.\t{attr_str}\n'
            )

        # --- 3. Generate feature lines for this specific ORF ---
        TIS, LTS, TTS, strand = (
            orf[f"TIS_coord{suffix}"],
            orf[f"LTS_coord{suffix}"],
            orf[f"TTS_coord{suffix}"],
            orf["strand"],
        )

        orf_fields = ["ORF_id", "ORF_type", "ribotie_score", "tis_transformer_score"]
        orf_attrs = {k: orf[k] for k in orf_fields if k in orf and orf[k] != "NA"}

        feature_definitions = []
        if TIS is not None:
            start_codon_stop = find_distant_exon_coord(TIS, 2, strand, exon_coord)
            feature_definitions.append(("start_codon", TIS, start_codon_stop))
        feature_definitions.append(("CDS", TIS, LTS))
        if TTS != -1:
            stop_codon_stop = find_distant_exon_coord(TTS, 2, strand, exon_coord)
            feature_definitions.append(("stop_codon", TTS, stop_codon_stop))

        for feature_type, start_coord, stop_coord in feature_definitions:
            if start_coord is None or stop_coord is None:
                continue

            parts, part_exons = transcript_region_to_exons(
                start_coord, stop_coord, strand, exon_coord
            )

            for i, (part_start, part_stop) in enumerate(np.array(parts).reshape(-1, 2)):
                feature_attrs = base_attrs | orf_attrs | {"exon_number": part_exons[i]}
                attr_str = (
                    "; ".join([f'{k} "{v}"' for k, v in feature_attrs.items()]) + ";"
                )
                gtf_lines.append(
                    f'{orf["seqname"]}\t{caller}\t{feature_type}\t{part_start}\t{part_stop}\t.\t{strand}\t.\t{attr_str}\n'
                )

    # --- Write all generated lines to the output GTF file ---
    with open(f"{out_prefix}.gtf", "w") as f:
        f.writelines(gtf_lines)
