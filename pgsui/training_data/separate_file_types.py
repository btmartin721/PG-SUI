import logging
import pprint
import shutil
from pathlib import Path


def copy_files(ext, dest, srcdir, alt_ext=None):
    """Copy files with a specific extension (or alternative extension) from a source directory to a destination directory, renaming the files to include the original subdirectory name.

    Args:
        ext (str): The primary file extension to match (e.g., ".txt").
        dest (str or Path): Destination directory where files will be copied.
        srcdir (str or Path): Source directory to recursively search for matching files.
        alt_ext (str, optional): An alternative extension to match in addition to "ext".
    """
    srcdir = Path(srcdir)
    outdir = Path(dest) / ext
    outdir.mkdir(parents=True, exist_ok=True)

    for resdir in srcdir.iterdir():
        if resdir.is_dir():
            for f in resdir.rglob("*"):
                if f.is_file():
                    if alt_ext:
                        if f.name.lower().endswith(ext) or f.name.lower().endswith(
                            alt_ext
                        ):
                            shutil.copy(f, outdir / f"{resdir.name}_{f.name}")
                    else:
                        if f.name.lower().endswith(ext):
                            shutil.copy(f, outdir / f"{resdir.name}_{f.name}")

    print(f"Copied {ext} files to {outdir}")


from pathlib import Path


def main():
    """Main function to count filetypes and copy files into organized folders."""
    logging.basicConfig(level=logging.INFO)
    logging.info("File Counts:\n")

    path = Path("training_data/extracted")
    extension_list = [
        ".phy",
        ".phylip",
        ".str",
        ".ustr",
        ".structure",
        ".loci",
        ".vcf",
        ".vcf.gz",
        ".nex",
        ".nexus",
    ]

    totals, overall_totals = count_filetypes(extension_list, path)

    [logging.info(f"{k}: {v}") for k, v in totals.items()]
    logging.info(f"Total files: {overall_totals}")

    outpath = Path("training_data", "collated")
    outpath.mkdir(parents=True, exist_ok=True)

    extension_list2 = [x.lstrip(".") for x in extension_list]

    for ext_lower, alt_ext in get_ext(extension_list2):
        copy_files(ext_lower, outpath, path, alt_ext=alt_ext)


def get_ext(extension_list):
    """Yield extension and alternative extension pairs for matching files.

    Args:
        extension_list (list of str): List of file extensions to match.

    Yields:
        tuple: (extension, alternative extension)
    """
    for ext in extension_list:
        # Define extension pairing logic
        ext_lower = ext.lower()
        alt_ext = None
        if ext_lower == "phy":
            alt_ext = "phylip"
        elif ext_lower == "str" or ext_lower == "ustr":
            alt_ext = "structure"
        elif ext_lower == "nex":
            alt_ext = "nexus"
        yield ext_lower, alt_ext


def count_filetypes(extension_list, mypath):
    """Count the number of files matching each extension in a directory and its subdirectories.

    Args:
        extension_list (list of str): List of file extensions to count (e.g., ['.txt', '.csv']).
        mypath (str or Path): Path to the directory to recursively search.

    Returns:
        tuple: (dictionary with counts per extension, overall total count of all matched files)
    """
    mypath = Path(mypath)

    totals = {}
    for ext in extension_list:
        file_count = sum(
            f.suffix.lower() == ext.lower() for f in mypath.rglob("*") if f.is_file()
        )
        totals[ext] = file_count

    overall_totals = sum(totals.values())

    return totals, overall_totals


if __name__ == "__main__":
    main()
