import json

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


def download(origin, doi, num):
    Pcurl2 = origin + "{}".format(doi)

    Bheaders2 = {"Content-Type": "application/zip"}

    r2 = requests.get(Pcurl2, headers=Bheaders2)

    with open("training_data/download/results{}.zip".format(num), "wb") as fout:
        fout.write(r2.content)


def search(origin, page, term, count, pbar):
    jsondata = curl(origin, page, term)

    if int(jsondata["count"]) == 0:
        return count
    else:
        for i in range(len(jsondata["_embedded"]["stash:datasets"])):
            tmp = jsondata["_embedded"]["stash:datasets"][i]["_links"][
                "stash:download"
            ]["href"].split("/")[-3:]

            doi = "/".join(tmp)

            download(origin, doi, count)

            count += 1

            pbar.update()

        page += 1

        count = search(origin, page, term, count, pbar)
        return count


def get_term_count(origin, page, term, total):
    jsondata = curl(origin, page, term)
    total = int(jsondata["total"])
    return total


def main():
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
    for term in searches:
        page = 1
        total += get_term_count(origin, page, term, total)

    for term in searches:
        page = 1

        with tqdm(desc="Downloading: ", total=total, leave=True, position=0) as pbar:
            counter = search(origin, page, term, counter, pbar)


if __name__ == "__main__":
    main()
