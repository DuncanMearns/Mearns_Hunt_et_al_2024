import shutil
from pathlib import Path


def files_to_folders(directory, glob, sort_func):
    directory = Path(directory)
    for src in directory.glob(glob):
        folder = sort_func(src)
        dst_directory = directory.joinpath(folder)
        if not dst_directory.exists():
            dst_directory.mkdir()
        dst = dst_directory.joinpath(src.name)
        shutil.move(src, dst)


if __name__ == "__main__":
    files_to_folders(r"D:\percomorph_analysis\tracking\eyes\n_multifasciatus",
                     "*.h5", lambda x: x.stem.split("_")[0])
