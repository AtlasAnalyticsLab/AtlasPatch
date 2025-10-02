"""Run CLAM tissue segmentation on a folder of WSIs and export masks.

Configuration is loaded from a JSON file referenced by `CONFIG_PATH` inside this
script. Update that constant to point at your preferred configuration file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.batch_process_utils import initialize_df

# Path to the configuration file (update as needed).
CONFIG_PATH = Path(__file__).with_name("clam_segmentation_config.json")


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def segment_slide(wsi: WholeSlideImage, seg_params, filter_params, mask_file=None):
    start_time = time.time()

    if mask_file is not None:
        wsi.initSegmentation(mask_file)
    else:
        wsi.segmentTissue(**seg_params, filter_params=filter_params)

    return time.time() - start_time


def run_segmentation(
    source: Path,
    save_dir: Path,
    mask_save_dir: Path,
    seg_params,
    filter_params,
    vis_params,
    use_default_params: bool = False,
    save_mask: bool = True,
    auto_skip: bool = False,
    mask_suffix: str = ".jpg",
):
    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(source / slide)]
    patch_params_stub = {"use_padding": True, "contour_fn": "four_pt"}

    df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params_stub)

    mask = df["process"] == 1
    process_stack = df[mask]
    total = len(process_stack)

    legacy_support = "a" in df.keys()
    if legacy_support:
        print("detected legacy segmentation csv file, legacy support enabled")
        df = df.assign(
            **{
                "a_t": np.full((len(df)), int(filter_params["a_t"]), dtype=np.uint32),
                "a_h": np.full((len(df)), int(filter_params["a_h"]), dtype=np.uint32),
                "max_n_holes": np.full((len(df)), int(filter_params["max_n_holes"]), dtype=np.uint32),
                "line_thickness": np.full((len(df)), int(vis_params["line_thickness"]), dtype=np.uint32),
            }
        )

    accumulated_seg_time = 0.0

    for i in tqdm(range(total)):
        df.to_csv(save_dir / "process_list_autogen.csv", index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, "slide_id"]
        print("\n\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
        print("processing {}".format(slide))

        df.loc[idx, "process"] = 0
        slide_id, _ = os.path.splitext(slide)
        mask_output_path = mask_save_dir / f"{slide_id}{mask_suffix}"

        if auto_skip and mask_output_path.is_file():
            print(f"{slide_id} mask already exists in destination location, skipped")
            df.loc[idx, "status"] = "already_exist"
            continue

        full_path = source / slide
        wsi_object = WholeSlideImage(str(full_path))

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}

            for key in vis_params.keys():
                if legacy_support and key == "vis_level":
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == "a_t":
                    old_area = df.loc[idx, "a"]
                    seg_level = df.loc[idx, "seg_level"]
                    scale = wsi_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == "seg_level":
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

        if current_vis_params["vis_level"] < 0:
            if len(wsi_object.level_dim) == 1:
                current_vis_params["vis_level"] = 0
            else:
                wsi = wsi_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params["vis_level"] = best_level

        if current_seg_params["seg_level"] < 0:
            if len(wsi_object.level_dim) == 1:
                current_seg_params["seg_level"] = 0
            else:
                wsi = wsi_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params["seg_level"] = best_level

        keep_ids = str(current_seg_params["keep_ids"])
        if keep_ids != "none" and len(keep_ids) > 0:
            str_ids = current_seg_params["keep_ids"]
            current_seg_params["keep_ids"] = np.array(str_ids.split(",")).astype(int)
        else:
            current_seg_params["keep_ids"] = []

        exclude_ids = str(current_seg_params["exclude_ids"])
        if exclude_ids != "none" and len(exclude_ids) > 0:
            str_ids = current_seg_params["exclude_ids"]
            current_seg_params["exclude_ids"] = np.array(str_ids.split(",")).astype(int)
        else:
            current_seg_params["exclude_ids"] = []

        w, h = wsi_object.level_dim[current_seg_params["seg_level"]]
        if w * h > 1e8:
            print(f"level_dim {w} x {h} is likely too large for successful segmentation, aborting")
            df.loc[idx, "status"] = "failed_seg"
            continue

        df.loc[idx, "vis_level"] = current_vis_params["vis_level"]
        df.loc[idx, "seg_level"] = current_seg_params["seg_level"]

        seg_time_elapsed = segment_slide(wsi_object, current_seg_params, current_filter_params)

        if save_mask:
            mask_img = wsi_object.visWSI(**current_vis_params)
            mask_img.save(mask_output_path)

        print(f"segmentation took {seg_time_elapsed} seconds")
        df.loc[idx, "status"] = "processed"
        accumulated_seg_time += seg_time_elapsed

    if total > 0:
        avg_seg_time = accumulated_seg_time / total
    else:
        avg_seg_time = 0.0

    df.to_csv(save_dir / "process_list_autogen.csv", index=False)
    print(f"average segmentation time in s per slide: {avg_seg_time}")

    return avg_seg_time


def main() -> None:
    config = load_config(CONFIG_PATH)

    source = Path(config["source"]).expanduser()
    save_dir = Path(config["save_dir"]).expanduser()
    mask_suffix = config.get("mask_suffix", ".jpg")
    auto_skip = bool(config.get("auto_skip", False))
    use_default_params = bool(config.get("use_default_params", False))
    mask_save_dir = save_dir / "masks"

    for path in (save_dir, mask_save_dir):
        ensure_dir(path)

    directories = {"source": source, "save_dir": save_dir, "mask_save_dir": mask_save_dir}

    for key, val in directories.items():
        print(f"{key} : {val}")

    seg_params = config.get(
        "seg_params",
        {"seg_level": -1, "sthresh": 8, "mthresh": 7, "close": 4, "use_otsu": False, "keep_ids": "none", "exclude_ids": "none"},
    )
    filter_params = config.get("filter_params", {"a_t": 100, "a_h": 16, "max_n_holes": 8})
    vis_params = config.get("vis_params", {"vis_level": -1, "line_thickness": 250})

    print({"seg_params": seg_params, "filter_params": filter_params, "vis_params": vis_params})

    run_segmentation(
        source=source,
        save_dir=save_dir,
        mask_save_dir=mask_save_dir,
        seg_params=seg_params,
        filter_params=filter_params,
        vis_params=vis_params,
        use_default_params=use_default_params,
        save_mask=True,
        auto_skip=auto_skip,
        mask_suffix=mask_suffix,
    )


if __name__ == "__main__":
    main()
