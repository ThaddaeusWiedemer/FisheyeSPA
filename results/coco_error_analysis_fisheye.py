import copy
import os
from argparse import ArgumentParser
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def intersect_slice_rect(
    slice_x: float,
    slice_y: float,
    slice_a1: float,
    slice_a2: float,
    rect_x: float,
    rect_y: float,
    rect_w: float,
    rect_h: float,
) -> bool:
    """Check whether a circle slice and a rectangle intersect.

    Args:
        slice_x (float): circle center x
        slice_y (float): circle center y
        slice_a1 (float): slice min angle (in degrees, counted clockwise from 0° pointing upwards)
        slice_a2 (float): slice max angle (in degrees, counted clockwise from 0° pointing upwards)
        rect_x (float): rectangle lower left x
        rect_y (float): rectangle lower left y
        rect_w (float): rectangle width
        rect_h (float): rectangle height
    
    Returns:
        bool: intersect
    """
    def between(x, l, u):
        return True if l <= x and x <= u else False

    # if the slice origin lies within the rectangle, the shapes trivially intersect
    if between(slice_x, rect_x, rect_x + rect_w) and between(slice_y, rect_y, rect_y + rect_h):
        return True

    # get angles from slice origin to all corners and
    def angle(x, y):
        return np.rad2deg((np.arctan2(x, y) % (2 * np.pi)))

    angle1 = angle(rect_x - slice_x, rect_y - slice_y)
    angle2 = angle(rect_x + rect_w - slice_x, rect_y - slice_y)
    angle3 = angle(rect_x - slice_x, rect_y + rect_h - slice_y)
    angle4 = angle(rect_x + rect_w - slice_x, rect_y + rect_h - slice_y)
    angle_min = min(angle1, angle2, angle3, angle4)
    angle_max = max(angle1, angle2, angle3, angle4)

    # if the min/max angle are both greater or both smaller then the slice
    # angles, the shapes don't intersect. Special attention needs to be payed at
    # retangles crossing the 360°/0°-line
    if rect_x - slice_x < 0 and rect_y + rect_h - slice_y > 0:
        # rectangle crosses 360°/0°-line
        if (angle_min + 360 > slice_a1 and angle_min + 360 > slice_a2 and angle_max > slice_a1
                and angle_max > slice_a2) and (angle_min < slice_a1 and angle_min < slice_a2
                                               and angle_max - 360 < slice_a1 and angle_max - 360 < slice_a2):
            return False
    else:
        # rectangle doesn't cross 0°-line
        if (angle_min > slice_a1 and angle_min > slice_a2 and angle_max > slice_a1
                and angle_max > slice_a2) or (angle_min < slice_a1 and angle_min < slice_a2 and angle_max < slice_a1
                                              and angle_max < slice_a2):
            return False

    # otherwise, the shapes intersect
    return True


def intersect_ring_rect(
    circ_x: float,
    circ_y: float,
    circ_ri: float,
    circ_ro: float,
    rect_x: float,
    rect_y: float,
    rect_w: float,
    rect_h: float,
) -> bool:
    """Checks whether a rectangle intersects a ring.

    Args:
        circ_x (float): ring center x
        circ_y (float): ring center y
        circ_ri (float): ring inner radius
        circ_ro (float): ring outer radius
        rect_x (float): rectangle lower left x
        rect_y (float): rectangle lower left y
        rect_w (float): rectangle width
        rect_h (float): rectangle height

    Returns:
        bool: intersect
    """

    # custom clamp function
    def clamp(x, l, u):
        return l if x < l else u if x > u else x

    # find point on rectangle closest to inner and outer circle
    # this is the same for both circles, since they share the same center
    closest_x = clamp(circ_x, rect_x, rect_x + rect_w)
    closest_y = clamp(circ_y, rect_y, rect_y + rect_h)
    # calculate distance to closest point
    dist = (circ_x - closest_x)**2 + (circ_y - closest_y)**2
    # compare to radii
    return circ_ri**2 < dist and dist < circ_ro**2


def get_coco_eval_df(cocoGt, cocoDt, imgIds, areas, metric_names, size_names):
    """Get the COCO evaluation results as pandas dataframe.

    Args:
        cocoGt (COCO): ground-truth annotations
        cocoDt (COCO): detections
        imgIds (COCO): images
        areas (list[float]): area sizes to use for small/medium/large evaluation
        metric_names (list[str]): names for the metrics in the data frame
        size_names (list[str]): names for different area sizes in evaluation

    Returns:
        pandas.DataFrame: long-form dataframe containing the columns 'metric': [metric names], 'size': [size names],
            'recall', and 'precision'
    """
    cocoEval = COCOeval(copy.deepcopy(cocoGt), copy.deepcopy(cocoDt), "bbox")
    cocoEval.params.imgIds = imgIds
    cocoEval.params.iouThrs = [0.75, 0.5, 0.1]
    cocoEval.params.maxDets = [100]
    if areas:
        cocoEval.params.areaRng = [
            [0**2, areas[2]],
            [0**2, areas[0]],
            [areas[0], areas[1]],
            [areas[1], areas[2]],
        ]
    cocoEval.evaluate()
    cocoEval.accumulate()
    # precision has shape [IoUs, recall thresholds, category, size, num detections]
    # we only have one category and don't impose a limit on detections, so we can reduce this to
    # [metric, recall threshold, size]
    ps = cocoEval.eval["precision"].squeeze()
    # get the value of the recall thresholds
    recThrs = cocoEval.params.recThrs
    # stack into one large array
    ps = np.vstack([ps, np.zeros((2, *ps.shape[1:]))])
    # fill in background and false negative errors
    ps[3, :, :] = ps[2, :, :] > 0
    ps[4, :, :] = 1.0
    # combine to pandas dataframe
    dfs = []
    for i, metric in enumerate(metric_names):
        df = pd.DataFrame(ps[i], columns=size_names)
        df["recall"] = recThrs
        df = df.melt(
            id_vars=["recall"],
            var_name="size",
            value_name="precision",
        )
        df = df[["size", "recall", "precision"]]
        df.insert(loc=0, column="metric", value=metric)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    return df


def plot_precision_recall(df: pd.DataFrame, outDir: str) -> None:
    """Plot all metrics as stacked precision-recall-curves with the areas annotated as APs.

    Generates 4 plots: 'overall.png' shows the results over all sizes, 'by_size.png' shows three individual graphs for
    each size. Both plots are also saved as .svg

    Args:
        df (pd.DataFrame): COCO evaluation result as Padas DataFrame
        outDir (str): path of directory to save images to
    """
    sb.set_context("talk")
    sb.set_style("whitegrid")

    line_style = ["--", "-", "-", "-", "-"]
    hatches = ["///", None, None, None]
    line_colors = ["#179C7D", "#179C7D", "#006E92", "#25BAE2"]
    fill_colors = ["#179C7D", "#179C7D", "#006E92", "#25BAE2"]
    text = [
        "IoU=0.75",
        "IoU=0.50",
        "if bbox was correct (IoU=0.1)",
        "if classification was correct",
    ]

    # first make a plot for overall info
    fig = plt.figure(figsize=(6, 4))

    _precision = np.zeros(len(df.loc[(df["size"] == "overall") & (df["metric"] == "AP@.75")]["recall"]))

    for i, metric in enumerate(["AP@.75", "AP@.50", "AP@.1", "BG"]):
        recall = df.loc[(df["size"] == "overall") & (df["metric"] == metric)]["recall"]
        precision = df.loc[(df["size"] == "overall") & (df["metric"] == metric)]["precision"]
        label = f"{np.mean(precision):.3f} | {text[i]}"

        if fill_colors[i] is None:
            plt.plot(
                recall,
                precision,
                color=line_colors[i],
                label=label,
                ls=line_style[i],
            )
        else:
            plt.plot(
                recall,
                precision,
                color=line_colors[i],
                label=label,
                ls=line_style[i],
            )
            plt.fill_between(recall,
                             precision,
                             _precision,
                             color=fill_colors[i],
                             alpha=0.3,
                             linewidth=0,
                             hatch=hatches[i])
        _precision = precision

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend(
        title="AP | setting",
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        frameon=False,
    )
    fig.savefig(outDir + "/overall.png", bbox_inches="tight")
    fig.savefig(outDir + "/overall.svg", bbox_inches="tight")
    plt.close(fig)

    # then make one figure for all sizes
    fig, ax = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    titles = [
        "Large Objects",
        "Medium-Sized Objects",
        "Small Objects",
    ]
    aps = []

    for s, size in enumerate(["large", "medium", "small"]):
        _precision = np.zeros(len(df.loc[(df["size"] == "overall") & (df["metric"] == "AP@.75")]["recall"]))
        aps.append([])

        for i, metric in enumerate(["AP@.75", "AP@.50", "AP@.1", "BG"]):
            recall = df.loc[(df["size"] == size) & (df["metric"] == metric)]["recall"]
            precision = df.loc[(df["size"] == size) & (df["metric"] == metric)]["precision"]
            aps[s].append(f"{np.mean(precision):.3f}")

            if fill_colors[i] is None:
                ax[s].plot(
                    recall,
                    precision,
                    color=line_colors[i],
                    label="placeholder",
                    ls=line_style[i],
                )
            else:
                ax[s].plot(
                    recall,
                    precision,
                    color=line_colors[i],
                    label="placeholder",
                    ls=line_style[i],
                )
                ax[s].fill_between(
                    recall,
                    precision,
                    _precision,
                    color=fill_colors[i],
                    alpha=0.3,
                    linewidth=0,
                    hatch=hatches[i],
                )
            _precision = precision

        ax[s].set_xlabel("recall")
        ax[s].set_title(titles[s])
        ax[s].set_xlim(0, 1.0)
    ax[0].set_ylabel("precision")
    plt.ylim(0, 1.0)
    L = plt.legend(
        title="AP L | AP M | AP S | setting",
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        frameon=False,
    )
    for i in range(4):
        L.get_texts()[i].set_text(f"{aps[0][i]} | {aps[1][i]} | {aps[2][i]} | {text[i]}")
    fig.savefig(outDir + "/by_size.png", bbox_inches="tight")
    fig.savefig(outDir + "/by_size.svg", bbox_inches="tight")
    plt.close(fig)


def plot_aps_size_distance_angle(ap_size: pd.DataFrame,
                                 ap_distance: pd.DataFrame,
                                 ap_angle: pd.DataFrame,
                                 outDir: str,
                                 alt: bool = False) -> None:
    """Plot the APs obtained from COCO eval results by object size, by distance to image center, and by angle from image
        center.

    Args:
        ap_size (pd.DataFrame): COCO evaluation result APs for different object sizes
        ap_distance (pd.DataFrame): COCO evaluation result APs for different object distances
        ap_angle (pd.DataFrame): COCO evaluation result APs for different angles
        outDir (str): path to output image directory
        alt (bool, optional): Plot alternate version of diagrams with more detail. Defaults to False.
    """
    sb.set_context("talk")
    sb.set_style("whitegrid")

    line_styles = ["--", "-", "-", "-"]
    hatches = ["///", None, None, None]
    colors = ["#179C7D", "#179C7D", "#006E92", "#25BAE2"]
    texts = [
        "IoU=0.75",
        "IoU=0.50",
        "if bbox was correct (IoU=0.1)",
        "if classification was correct",
    ]

    fig = plt.figure(figsize=(18, 4))
    ax = []
    ax.append(plt.subplot(131))
    ax.append(plt.subplot(132))
    ax.append(plt.subplot(133, projection="polar"))

    def cycle(l):
        return np.append(l, l[0])

    # the metrics have to be plotted in reversed order for the bar charts to stack properly
    for i, metric in enumerate(reversed(["AP@.75", "AP@.50", "AP@.1", "BG"])):
        # plot simple bar chart for different sizes
        ax[0].bar(
            ap_size["size"].unique(),
            ap_size[ap_size["metric"] == metric]["ap"],
            color=colors[3 - i],
            width=16 / 25,
            hatch=hatches[3 - i],
        )

        # plot either bar char for different distances or grouped bar chart for different sizes for each distance
        if alt:
            for j, size in enumerate(ap_distance["size"].unique()):
                _labels = ap_distance["distance"].unique()
                x = np.arange(len(_labels))
                ax[1].bar(
                    x - 0.3 + 0.2 * j,
                    ap_distance.loc[(ap_distance["metric"] == metric) & (ap_distance["size"] == size)]["ap"],
                    color=colors[3 - i],
                    hatch=hatches[3 - i],
                    width=0.15,
                )
        else:
            ax[1].bar(
                ap_distance["distance"].unique(),
                ap_distance.loc[(ap_distance["metric"] == metric) & (ap_distance["size"] == "overall")]["ap"],
                color=colors[3 - i],
                hatch=hatches[3 - i],
            )

        # plot polar plot to show AP per angle
        ax[2].plot(
            cycle(ap_angle["angle"].unique()) / 180 * np.pi,
            cycle(ap_angle[ap_angle["metric"] == metric]["ap"].to_numpy()),
            color=colors[3 - i],
            ls=line_styles[3 - i],
            label=texts[3 - i],
        )

    ax[0].set_ylim(0, 1.0)
    ax[1].set_ylim(0, 1.0)
    ax[2].set_ylim(0, 1.0)
    ax[0].set_title("APs by Object Size")
    ax[1].set_title("APs by Distance to Center")
    ax[2].set_title("APs by Angle")
    if alt:
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(_labels)
    ax[0].set_ylabel("AP")
    ax[1].set_yticklabels([])
    ax[2].set_theta_zero_location("N")
    ax[2].set_theta_direction(-1)
    handles, labels = ax[2].get_legend_handles_labels()
    order = [3, 2, 1, 0]
    plt.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        title="setting",
        bbox_to_anchor=(1.2, 0.5),
        loc="center left",
        frameon=False,
    )
    if alt:
        fig.savefig(
            outDir + "/ap_size_distance_angle_alt.png",
            bbox_inches="tight",
        )
        fig.savefig(
            outDir + "/ap_size_distance_angle_alt.svg",
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            outDir + "/ap_size_distance_angle.png",
            bbox_inches="tight",
        )
        fig.savefig(
            outDir + "/ap_size_distance_angle.svg",
            bbox_inches="tight",
        )
    plt.close(fig)


def analyze_results(
        res_file,
        ann_file,
        out_dir,
        extraplots=None,
        areas=None,
        center=(400, 300),
):
    if areas:
        assert (len(areas) == 3), "3 integers should be specified as areas, \
            representing 3 area regions"

    directory = os.path.dirname(out_dir + "/")
    if not os.path.exists(directory):
        print(f"-------------create {out_dir}-----------------")
        os.makedirs(directory)

    metrics = ["AP@.75", "AP@.50", "AP@.1", "BG", "FN"]
    sizes = ["overall", "small", "medium", "large"]
    distances = range(64, 321, 64)
    angles = range(0, 360, 5)

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)
    imgIds = cocoGt.getImgIds()
    # get overall evaluation
    df = get_coco_eval_df(cocoGt, cocoDt, imgIds, areas, metrics, sizes)
    # make plot for precision-recall-curve by size
    plot_precision_recall(df, out_dir)

    # get APs by size
    ap_size = pd.DataFrame(columns=["size", "metric", "ap"])
    for size in ["overall", "large", "medium", "small"]:
        for metric in metrics:
            ap_size = ap_size.append(
                {
                    "size": size,
                    "metric": metric,
                    "ap": np.mean(df.loc[(df["size"] == size) & (df["metric"] == metric)]["precision"]),
                },
                ignore_index=True,
            )

    # get APs by distance
    ap_distance = pd.DataFrame(columns=["distance", "size", "metric", "ap"])
    for d in distances:
        # only consider detections that intersect the current ring
        dt = copy.deepcopy(cocoDt)
        dt_anns = dt.dataset["annotations"]
        select_dt_anns = []
        for ann in dt_anns:
            if intersect_ring_rect(*center, d - 64, d, *ann["bbox"]):
                select_dt_anns.append(ann)
        dt.dataset["annotations"] = select_dt_anns
        dt.createIndex()
        # only consider ground-truths that intersect the current ring
        gt = copy.deepcopy(cocoGt)
        for idx, ann in enumerate(gt.dataset["annotations"]):
            if not intersect_ring_rect(*center, d - 64, d, *ann["bbox"]):
                gt.dataset["annotations"][idx]["ignore"] = 1
                gt.dataset["annotations"][idx]["iscrowd"] = 1
        # evaluate and compute APs
        df = get_coco_eval_df(gt, dt, imgIds, areas, metrics, sizes)
        for size in ["overall", "large", "medium", "small"]:
            for metric in metrics:
                ap_distance = ap_distance.append(
                    {
                        "distance": f"{d}",
                        "size": size,
                        "metric": metric,
                        "ap": np.mean(df.loc[(df["size"] == size) & (df["metric"] == metric)]["precision"]),
                    },
                    ignore_index=True,
                )

    # get APs by angle
    ap_angle = pd.DataFrame(columns=["angle", "metric", "ap"])
    for a in angles:
        # only consider detections that intersect the current slice
        dt = copy.deepcopy(cocoDt)
        dt_anns = dt.dataset["annotations"]
        select_dt_anns = []
        for ann in dt_anns:
            if intersect_slice_rect(*center, a, a + 5, *ann["bbox"]):
                select_dt_anns.append(ann)
        dt.dataset["annotations"] = select_dt_anns
        dt.createIndex()
        # only consider ground-truths that intersect the current slice
        gt = copy.deepcopy(cocoGt)
        for idx, ann in enumerate(gt.dataset["annotations"]):
            if not intersect_slice_rect(*center, a, a + 5, *ann["bbox"]):
                gt.dataset["annotations"][idx]["ignore"] = 1
                gt.dataset["annotations"][idx]["iscrowd"] = 1
        # evaluate and compute APs
        df = get_coco_eval_df(gt, dt, imgIds, areas, metrics, sizes)
        for metric in metrics:
            ap_angle = ap_angle.append(
                {
                    "angle": a,
                    "metric": metric,
                    "ap": np.mean(df.loc[(df["size"] == "overall") & (df["metric"] == metric)]["precision"]),
                },
                ignore_index=True,
            )

    # make plot for APs by size, distance, angle
    plot_aps_size_distance_angle(ap_size, ap_distance, ap_angle, out_dir)
    plot_aps_size_distance_angle(ap_size, ap_distance, ap_angle, out_dir, alt=True)


def main():
    parser = ArgumentParser(description="COCO Error Analysis Tool")
    parser.add_argument("result", help="result file (json format) path")
    parser.add_argument("out_dir", help="dir to save analyze result images")
    parser.add_argument(
        "--ann",
        default="data/coco/annotations/instances_val2017.json",
        help="annotation file path",
    )
    # parser.add_argument(
    #     '--types', type=str, nargs='+', default=['bbox'], help='result types')
    parser.add_argument(
        "--extraplots",
        action="store_true",
        help="export extra bar/stat plots",
    )
    parser.add_argument(
        "--areas",
        type=int,
        nargs="+",
        default=[1024, 9216, 10000000000],
        help="area regions",
    )
    args = parser.parse_args()
    analyze_results(
        args.result,
        args.ann,
        # args.types,
        out_dir=args.out_dir,
        extraplots=args.extraplots,
        areas=args.areas,
    )


if __name__ == "__main__":
    main()
