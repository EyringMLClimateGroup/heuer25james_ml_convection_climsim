from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
import multiprocessing as mp
# import functools

# Define the model/repository name and the pattern
res="high-res"
# repo_id = "LEAP/ClimSim_low-res"  # Replace with the actual repository name
repo_id = f"LEAP/ClimSim_{res}"  # Replace with the actual repository name

## ------------ Exemplary for first year ------------
patterns = [f"*E3SM-MMF.ml*.0001-*-{i:02d}-*.nc" for i in range(1,2)]  # Replace with your desired pattern
print(patterns)

local_path = f'/scratch/b/b309215/LEAP/ClimSim_{res}/'


if __name__ == '__main__':
    print('cpu_count: ', mp.cpu_count())

    snapshot_download(repo_id=repo_id, allow_patterns=patterns, repo_type="dataset", local_dir=local_path, max_workers=mp.cpu_count())
