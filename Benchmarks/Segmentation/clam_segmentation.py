"""Run CLAM tissue segmentation on a folder of WSIs and export masks."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.batch_process_utils import initialize_df


def segment_slide(wsi: WholeSlideImage, seg_params, filter_params, mask_file=None):
    start_time = time.time()

    if mask_file is not None:
        wsi.initSegmentation(mask_file)
    else:
        wsi.segmentTissue(**seg_params, filter_params=filter_params)

    return time.time() - start_time


def run_segmentation(
    source: str,
    save_dir: str,
    mask_save_dir: str,
    seg_params,
    filter_params,
    vis_params,
    use_default_params: bool = False,
    save_mask: bool = True,
    auto_skip: bool = False,
    process_list: str | None = None,
    mask_suffix: str = ".jpg",
):
    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    patch_params_stub = {"use_padding": True, "contour_fn": "four_pt"}

    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params_stub)
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params_stub)

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
        df.to_csv(os.path.join(save_dir, "process_list_autogen.csv"), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, "slide_id"]
        print("\n\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
        print("processing {}".format(slide))

        df.loc[idx, "process"] = 0
        slide_id, _ = os.path.splitext(slide)
        mask_output_path = os.path.join(mask_save_dir, slide_id + mask_suffix)

        if auto_skip and os.path.isfile(mask_output_path):
            print("{} mask already exists in destination location, skipped".format(slide_id))
            df.loc[idx, "status"] = "already_exist"
            continue

        full_path = os.path.join(source, slide)
        wsi_object = WholeSlideImage(full_path)

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
            print("level_dim {} x {} is likely too large for successful segmentation, aborting".format(w, h))
            df.loc[idx, "status"] = "failed_seg"
            continue

        df.loc[idx, "vis_level"] = current_vis_params["vis_level"]
        df.loc[idx, "seg_level"] = current_seg_params["seg_level"]

        seg_time_elapsed = segment_slide(wsi_object, current_seg_params, current_filter_params)

        if save_mask:
            mask_img = wsi_object.visWSI(**current_vis_params)
            mask_img.save(mask_output_path)

        print("segmentation took {} seconds".format(seg_time_elapsed))
        df.loc[idx, "status"] = "processed"
        accumulated_seg_time += seg_time_elapsed

    if total > 0:
        avg_seg_time = accumulated_seg_time / total
    else:
        avg_seg_time = 0.0

    df.to_csv(os.path.join(save_dir, "process_list_autogen.csv"), index=False)
    print("average segmentation time in s per slide: {}".format(avg_seg_time))

    return avg_seg_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLAM tissue segmentation and export masks")
    parser.add_argument("--source", type=str, required=True, help="Path to folder containing raw WSI image files")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--auto_skip", action="store_true", help="Skip slides whose mask already exists")
    parser.add_argument("--preset", default=None, type=str, help="CSV of default segmentation and filter parameters")
    parser.add_argument("--process_list", type=str, default=None, help="CSV of slides to process with parameters")
    parser.add_argument("--mask_suffix", type=str, default=".jpg", help="Extension for saved masks (e.g. .jpg/.png)")
    parser.add_argument("--use_default_params", action="store_true", help="Use default parameters for all slides")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    save_dir = args.save_dir
    mask_save_dir = os.path.join(save_dir, "masks")

    if args.process_list:
        process_list_path = os.path.join(save_dir, args.process_list)
    else:
        process_list_path = None

    directories = {"source": args.source, "save_dir": save_dir, "mask_save_dir": mask_save_dir}

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key != "source":
            os.makedirs(val, exist_ok=True)

    seg_params = {"seg_level": -1, "sthresh": 8, "mthresh": 7, "close": 4, "use_otsu": False,
                  "keep_ids": "none", "exclude_ids": "none"}
    filter_params = {"a_t": 100, "a_h": 16, "max_n_holes": 8}
    vis_params = {"vis_level": -1, "line_thickness": 250}

    if args.preset:
        preset_df = pd.read_csv(os.path.join("presets", args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]
        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]
        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

    parameters = {"seg_params": seg_params, "filter_params": filter_params, "vis_params": vis_params}
    print(parameters)

    run_segmentation(
        **directories,
        **parameters,
        use_default_params=args.use_default_params,
        save_mask=True,
        auto_skip=args.auto_skip,
        process_list=process_list_path,
        mask_suffix=args.mask_suffix,
    )
