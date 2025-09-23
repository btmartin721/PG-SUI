import json
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def curl(origin, page, term):

    Bheaders = {"Content-Type": "application/json"}

    PCurl = origin + "search?page={}&per_page=100&q={}".format(page, term)

    # GET request
    r = requests.get(PCurl, headers=Bheaders)

    # create json object
    jsondata = json.loads(r.text)

    return jsondata


def get_metadata(origin, doi, title, keywords, num, abstract, authors):

    root = f"training_data/metadata/results{num}"
    Path(root).mkdir(parents=True, exist_ok=True)

    kw = ";".join(keywords)
    title = title.replace("\n", " ")

    header = "dataset,doi,title,keywords"
    record = f"results{num},{doi},{title},{kw}"

    try:
        with open(f"{root}/metadata.txt", "w") as fout:
            fout.write(f"{header}\n")
            fout.write(f"{record}")

    except UnicodeEncodeError:
        with open(f"{root}/metadata.txt", "w", encoding="utf-8") as fout:
            fout.write(f"{header}\n")
            fout.write(f"{record}")


def search(origin, page, term, count, pbar, total, df):
    jsondata = curl(origin, page, term)

    if int(jsondata["count"]) == 0:
        return count, df
    else:
        for i in range(len(jsondata["_embedded"]["stash:datasets"])):

            pbar.update()

            record_num = f"results{count}"

            try:
                tmp = jsondata["_embedded"]["stash:datasets"][i]["_links"][
                    "stash:download"
                ]["href"].split("/")[-3:]

            except KeyError:
                count += 1
                pbar.update()
                col_list = list()
                col_list.append(record_num)
                col_list.append(pd.NA)
                col_list.append(pd.NA)
                col_list.append(pd.NA)
                col_list.append(pd.NA)
                col_list.extend([pd.NA, pd.NA, pd.NA, pd.NA, pd.NA])

                cols = ["DatasetID", "DOI", "Title", "Keywords", "Abstract"]
                cols.extend(
                    [
                        "firstName_1",
                        "lastName_1",
                        "email_1",
                        "affiliation_1",
                        "affiliationROR_1",
                    ]
                )

                dfrow = pd.DataFrame([col_list], columns=cols)
                df = pd.concat([df, dfrow], ignore_index=True)
                continue

            doi = "/".join(tmp)
            doi = doi.replace("%3A", ":")
            doi = doi.replace("%2", "/")

            try:
                title = jsondata["_embedded"]["stash:datasets"][i]["title"]
            except KeyError:
                title = pd.NA

            try:
                keywords = jsondata["_embedded"]["stash:datasets"][i]["keywords"]
                kw = ",".join(keywords)

            except KeyError:
                kw = pd.NA

            try:
                abstract = jsondata["_embedded"]["stash:datasets"][i]["abstract"]
            except KeyError:
                abstract = pd.NA

            try:
                tmp_authors = jsondata["_embedded"]["stash:datasets"][i]["authors"]

                authors_keys = list()
                authors_values = list()
                all_author_keys = list()
                all_author_values = list()
                for cnt, auth in enumerate(tmp_authors, start=1):
                    auth_keys = list(auth.keys())
                    auth_vals = list(auth.values())

                    authors_keys.append(([f"{x}_{cnt}" for x in auth_keys]))
                    authors_values.append(auth_vals)

                # Flatten 2d list to 1d
                all_author_keys = sum(authors_keys, [])
                all_author_values = sum(authors_values, [])

            except KeyError:
                authors_keys = [
                    "firstName_1",
                    "lastName_1",
                    "email_1",
                    "affiliation_1",
                    "affiliationROR_1",
                ]
                authors_values = [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]

            cols = ["DatasetID", "DOI", "Title", "Keywords", "Abstract"]
            cols.extend(all_author_keys)

            # Code adapted from:
            # https://stackoverflow.com/questions/51720306/creating-a-dataframe-from-multiple-lists-with-list-names-as-column-names

            col_list = list()
            col_list.append(record_num)
            col_list.append(doi)
            col_list.append(title)
            col_list.append(kw)
            col_list.append(abstract)
            col_list.extend(all_author_values)

            dfrow = pd.DataFrame([col_list], columns=cols)
            df = pd.concat([df, dfrow], ignore_index=True)

            count += 1

        page += 1

        count, df = search(origin, page, term, count, pbar, total, df)

        return count, df


def get_term_count(origin, page, term, count, total):
    jsondata = curl(origin, page, term)
    total = int(jsondata["total"])
    return total


origin = "https://datadryad.org/api/v2/"

searches = [
    "bfd",
    "ddrad*",
    "ipyrad",
    "pyrad",
    "species%20delimitation",
    "unlinked%20loci",
    "unlinked%20snps",
    "population%20genomic",
    "population%20genetic",
    "phylogenomics",
    "phylogenetics",
    "phylogeography",
    "phylogenetic%20analysis",
    "phylogenetic%20tree",
    "phylogenetic%20inference",
    "snp%20data",
    "RADseq*",
    "RAD-seq*",
]
counter = 1
total = 0
df = pd.DataFrame()
for term in searches:
    page = 1
    total += get_term_count(origin, page, term, counter, total)

print(f"Parsing {total} datasets...")

with tqdm(desc="Fetching Metadata: ", total=total, position=0, leave=True) as pbar:

    for term in searches:
        page = 1
        counter, df = search(origin, page, term, counter, pbar, total, df)

Path("training_data/metadata").mkdir(parents=True, exist_ok=True)
df.to_csv("training_data/metadata/metadata.csv", sep=",", header=True, index=False)

contains_list = [
    "ddrad",
    "radseq",
    "rad sequencing",
    "bfd",
    "snp",
    "snps",
    "genomic",
    "genomics",
    "genome",
    "genomes",
    "pyrad",
    "ipyrad",
    "species delimitation",
    "stacks",
    "phylogenomic",
    "phylogenomics",
    "population genomic",
    "population genomics",
    "landscape genomics",
    "hybrid zone",
    "contact zone",
    "introgression",
    "introgress",
    "structure",
    "admixture",
]
contains_list = "|".join(contains_list)

genomic_df = df[
    df.apply(
        lambda r: r.str.lower().str.contains(contains_list, case=False).any(), axis=1
    )
]

genomic_df.to_csv(
    "training_data/metadata/metadata_genomic_data.csv",
    sep=",",
    header=True,
    index=False,
)

print(f"Found {len(genomic_df.index)} genomic datasets")
