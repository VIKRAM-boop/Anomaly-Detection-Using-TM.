# Running this pipeline on Google Colab

The scripts are ~100KB; the data is ~4.8GB. Do not upload the data — the
pipeline regenerates it from the Kaggle source in about 10 minutes.

## Cell 1 — install dependencies

```python
!pip install -q tmu imbalanced-learn kagglehub
!pip install -q "numpy>=2.0,<2.3" --force-reinstall
```

The numpy pin matters: numba requires numpy<2.3, and an unpinned install
breaks the tmu import.

## Cell 2 — patch tmu for NumPy 2.x

```python
import re, pathlib, tmu
root = pathlib.Path(tmu.__file__).parent
patched = 0
for f in root.rglob("*.py"):
    src = f.read_text()
    new = re.sub(r"~0\b", "np.uint32(0xFFFFFFFF)", src)
    if new != src:
        f.write_text(new); patched += 1
print(f"patched {patched} files")
```

NumPy 2.x raises `OverflowError` on `np.uint32(-1)` where 1.x silently
wrapped. tmu relies on the old behaviour, so the import fails without this.

**Restart the runtime after this cell** (Runtime > Restart session).

## Cell 3 — get the code

```python
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>
```

Or upload the eight files in `scripts/` directly via the file browser.

## Cell 4 — download the raw dataset

```python
import kagglehub, shutil, pathlib
src = pathlib.Path(kagglehub.dataset_download("mrwellsdavid/unsw-nb15"))
dst = pathlib.Path("raw_data_full"); dst.mkdir(exist_ok=True)
for n in ["UNSW-NB15_1.csv","UNSW-NB15_2.csv","UNSW-NB15_3.csv","UNSW-NB15_4.csv",
          "NUSW-NB15_features.csv"]:
    shutil.copy(src/n, dst/n)
print(sorted(p.name for p in dst.iterdir()))
```

149MB download, no Kaggle login needed.

## Cell 5 — point the scripts at Colab paths

Two scripts read parquet from a local scratch directory used to avoid a
Dropbox sync issue on the original machine. On Colab that directory does
not exist:

```python
!sed -i 's|Path("/tmp/tmwork2")|find_dir("data") / "temporal"|' scripts/01c_preprocess_temporal.py
!sed -i 's|PARQUET = Path("/tmp/tmwork2")|PARQUET = find_dir("data") / "temporal"|' scripts/07_tune_baselines.py
```

## Cell 6 — run the pipeline

```python
%cd scripts
!python 00_build_temporal_windows.py      # ~3 min
!python 01c_preprocess_temporal.py        # ~6 min
!python 02c_train_with_scores.py medium 42   # ~30 min
!python 03_calibrate.py
```

Optional extras:

```python
!python 02c_train_with_scores.py short 42   # for the ensemble
!python 02c_train_with_scores.py long 42
!python 08_multiscale_ensemble.py
!python 07_tune_baselines.py                # RF + DT, ~40 min
```

## Notes

- **Disk**: Colab gives ~100GB, so the ~4.8GB generated is fine.
- **Session limits**: free-tier Colab disconnects after ~90 minutes of
  browser inactivity and caps sessions at ~12 hours. A disconnect can wipe
  the runtime entirely, losing installed packages and generated data. Save
  anything you need to Drive as you go.
- **Speed**: free-tier Colab gives 2 vCPUs. tmu here runs on CPU, so
  expect it to be no faster than a local machine, possibly slower.
- **Persisting results**: mount Drive and copy outputs, e.g.

  ```python
  from google.colab import drive; drive.mount('/content/drive')
  !cp -r ../data/scores_* /content/drive/MyDrive/
  ```
