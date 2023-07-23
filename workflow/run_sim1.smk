import os
import sys
import subprocess
import datetime
from dateutil import parser
from os import listdir
from os.path import isfile, join
from collections import OrderedDict
import pandas as pd
import numpy as np
from glob import glob

###################################################
# Parse configfile 

# make sure required params are set
if not config:
    raise Exception("Please provide a configuration file with --configfile")

required_config_params = ["raw_reads", 
                    "out_dir", 
                    "sample_metadata", 
                    "reference",
                    "bwa_index"]

missing_params = [param for param in required_config_params if param not in config]
if missing_params:
    raise Exception(f"Missing required configuration parameters: {', '.join(missing_params)}")

# flags for the optional params
use_amplicons = "amplicons" in config and bool(config["amplicons"].strip())
use_primers = "primers" in config and bool(config["primers"].strip())
use_adapters = "adapter_fasta" in config and bool(config["adapter_fasta"].strip())

###################################################
# Parse input metadata to get list of samples

lst = os.listdir(config["raw_reads"])
files = [f for f in lst if os.path.isfile(os.path.join(config["raw_reads"], f)) or os.path.islink(os.path.join(config["raw_reads"], f))]
files.sort()

SAMPLES = []
for i in files:
    SAMPLES.append(i.split("_")[0])
SAMPLES = sorted(list(set(SAMPLES)))
seqdf = pd.DataFrame.from_dict({"Sample": SAMPLES})
seqdf['Sample'] = seqdf['Sample'].astype(str)

# get metadata
metadata_all = pd.read_csv(config["sample_metadata"], header=0, sep="\t")
metadata_all['Sample']=metadata_all['Sample'].astype(str)
metadata = seqdf.merge(metadata_all, on='Sample', how='left', indicator=True)

# Remove excluded samples (regardless of reason for exclusion)
# excluded = pd.read_csv(excluded_samples, header=0, sep="\t")
# excluded['Sample']=excluded['Sample'].astype(str)
# metadata = metadata[~metadata.Sample.isin(excluded.Sample)].reset_index()

###################################################
# Define inputs for Snakemake
# Probably a better way of doing this....

# define inputs
base_outputs = ["{out_dir}/cleaned_reads/{batch}/{sample}_1.fq.gz",
                "{out_dir}/cleaned_reads/{batch}/{sample}_2.fq.gz",
                "{out_dir}/samtools_stats/{batch}/{sample}.stats",
                "{out_dir}/mosdepth/{batch}/{sample}.mosdepth.summary.txt",
                "{out_dir}/mpileup/{batch}/{sample}.pileup.gz"]

# Conditionally append to base_outputs
if use_amplicons:
    base_outputs.append("{out_dir}/mosdepth/{batch}/{sample}.regions.bed.gz")
if use_primers:
    base_outputs.append("{out_dir}/ivar_trim/{batch}/{sample}.sorted.bam")

# Expand outputs with batch and sample
expanded_outputs = expand(base_outputs, zip,
                          batch=metadata[metadata["_merge"] == "both"]["Batch"],
                          sample=metadata[metadata["_merge"] == "both"]["Sample"],
                          allow_missing=True)

# Add the output directory and unique batch to the outputs
final_outputs = expand(expanded_outputs, out_dir=config["out_dir"], 
                       batch=pd.unique(metadata[metadata["_merge"] == "both"]["Batch"]))

##################################################
# SnakeMake workflow 

rule all:
    input:
        final_outputs


# NOTE: Temporarily using the full sorted bam, NOT primer-trimmed
rule get_pileups:
    input:
        "{out_dir}/aligned/{batch}/{sample}.sorted.bam",
        config["reference"]
    output:
        "{out_dir}/mpileup/{batch}/{sample}.pileup.gz"
    log:
        "{out_dir}/logs/{batch}/{sample}_mpileup.log"
    conda:
        "envs/bwa.yml"
    shell:
        "mkdir -p $(dirname {output})/ && "
        "samtools mpileup -aa -A -d 600000 -Q 20 -q 0 -B -f {input[1]} {input[0]} | gzip > {output} 2> {log} ; "

rule trim_primer_quality:
    input:
        "{out_dir}/aligned/{batch}/{sample}.sorted.bam",
        config["primers"]
    output:
        "{out_dir}/ivar_trim/{batch}/{sample}.sorted.bam",
        "{out_dir}/ivar_trim/{batch}/{sample}_ivar.log"
    params:
        prefix="{out_dir}/ivar_trim/{batch}/{sample}"
    log:
        "{out_dir}/logs/{batch}/{sample}_ivar-trim.log"
    conda:
        "envs/ivar.yml"
    shell:
        "mkdir -p $(dirname {output[0]})/ && "
        "ivar trim -b {input[1]} -p {params.prefix} -i {input[0]} > {output[1]} 2> {log} ; "
        "samtools sort -T {wildcards.sample} -o {output[0]} {params.prefix}.bam &&"
        "samtools index {output[0]} >> {log} 2>&1"

rule mosdepth_with_amplicons:
    input:
        f"{config['out_dir']}/aligned/{{batch}}/{{sample}}.sorted.bam", 
        config["amplicons"]
    output:
        "{out_dir}/mosdepth/{batch}/{sample}.mosdepth.global.dist.txt",
        "{out_dir}/mosdepth/{batch}/{sample}.regions.bed.gz",
        "{out_dir}/mosdepth/{batch}/{sample}.mosdepth.summary.txt"
    params:
        prefix="{out_dir}/mosdepth/{batch}/{sample}"
    log:
        "{out_dir}/logs/{batch}/{sample}_mosdepth.log"
    conda:
        "envs/mosdepth.yml"
    shell:
        "mkdir -p $(dirname {output[0]})/ && "
        "mosdepth --by {input[1]} --no-per-base -T 10,30,100 --fast-mode {params.prefix} {input[0]} > {log} 2>&1"

rule mosdepth_without_amplicons:
    input:
        f"{config['out_dir']}/aligned/{{batch}}/{{sample}}.sorted.bam",
    output:
        "{out_dir}/mosdepth/{batch}/{sample}.mosdepth.global.dist.txt",
        "{out_dir}/mosdepth/{batch}/{sample}.mosdepth.summary.txt"
    params:
        prefix="{out_dir}/mosdepth/{batch}/{sample}"
    log:
        "{out_dir}/logs/{batch}/{sample}_mosdepth.log"
    conda:
        "envs/mosdepth.yml"
    shell:
        "mkdir -p $(dirname {output[0]})/ && "
        "mosdepth --no-per-base --fast-mode {params.prefix} {input[0]} > {log} 2>&1"

# set rule precedence for mosdepth 
ruleorder: mosdepth_with_amplicons > mosdepth_without_amplicons

rule samtools_stats:
    input:
        "{out_dir}/aligned/{batch}/{sample}.sorted.bam"
    output:
        "{out_dir}/samtools_stats/{batch}/{sample}.stats"
    log:
        "{out_dir}/logs/{batch}/{sample}_samstats.log"
    conda:
        "envs/bwa.yml"
    shell:
        "mkdir -p $(dirname {output})/ && "
        "samtools stats {input} 2> {log} > {output}"

rule align_reads:
    input:
        expand("{out_dir}/cleaned_reads/{{batch}}/{{sample}}_{num}.fq.gz", out_dir=config["out_dir"], num=[1,2])
    params:
        index=config["bwa_index"]
    output:
        "{out_dir}/aligned/{batch}/{sample}.sorted.bam"
    log:
        "{out_dir}/logs/{batch}/{sample}_bwa.log"
    conda:
        "envs/bwa.yml"
    shell:
        "mkdir -p $(dirname {output})/ && "
        "bwa mem {params.index} {input[0]} {input[1]} 2> {log} | samtools view -F 4 -Sb 2>> {log} | samtools sort -T {wildcards.sample} -o {output}  2>> {log} && "
        "samtools index {output} >> {log} 2>&1"

rule fastp:
    input:
        expand("{raw_reads}/{{sample}}_R{num}.fq.gz", raw_reads=config["raw_reads"], num=[1,2])
    output:
        "{out_dir}/cleaned_reads/{batch}/{sample}_1.fq.gz",
        "{out_dir}/cleaned_reads/{batch}/{sample}_2.fq.gz",
        "{out_dir}/fastp/{batch}/{sample}_fastp.html",
        "{out_dir}/fastp/{batch}/{sample}_fastp.json"
    params:
        adapter=config["adapter_fasta"] if use_adapters else ""
    log:
        "{out_dir}/logs/{batch}/{sample}_fastp.log"
    conda:
        "envs/fastp.yml"
    shell:
        """
        mkdir -p $(dirname {output[0]})/ && 
        mkdir -p $(dirname {output[2]})/ && 
        fastp_cmd="fastp -i {input[0]} -I {input[1]} -o {output[0]} -O {output[1]} -h {output[2]} -j {output[3]} 
        --correction --overrepresentation_analysis -g -l 50 -5 1 -3 1 
        --detect_adapter_for_pe"
        
        if [ "{params.adapter}" != "" ]; then
            fastp_cmd+=" --adapter_fasta {params.adapter}"
        fi
        
        $fastp_cmd > {log} 2>&1
        """

