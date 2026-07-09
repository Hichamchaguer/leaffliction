import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def get_files(__dir: str = "leaves/") -> dict:
    """Scan directory and count files in subdirectories grouped by prefix.

    Args:
        __dir (str, optional): Directory to scan. Defaults to "leaves/".

    Returns:
        dict: Nested dict with prefix as key and subdir:count pairs as values.
    """
    subs = {}
    for elem in os.listdir(__dir):
        path = os.path.join(__dir, elem)
        if not os.path.isdir(path):
            continue
        sub_d = elem.split("_")[0]
        count = len(os.listdir(path))
        if sub_d not in subs:
            subs[sub_d] = {}
        subs[sub_d][elem] = count
    return subs


def plot_pie(ax: plt.axes, data: list) -> None:
    """Plot a pie chart with the largest segment in white text.

    Args:
        ax (plt.axes): Matplotlib axes to plot on.
        data (list): Values for pie segments.
    """
    max_indx = data.index(max(data))
    _, text, autotexts = ax.pie(data, autopct='%1.1f%%')
    for i, val in enumerate(autotexts):
        if i == max_indx:
            val.set_color("#fff")
            break


def plot_bar(ax: plt.axes, x: list, y: list) -> None:
    """Plot a bar chart with tab10 color palette.

    Args:
        ax (plt.axes): Matplotlib axes to plot on.
        x (list): Category labels for x-axis.
        y (list): Values for bar heights.
    """
    ax.bar(x, y, color=plt.cm.tab10.colors[:len(x)])


def main():
    try:
        assert len(sys.argv) == 2, "Need only the directory path"
        leaves = get_files(sys.argv[1])
        for key, val in leaves.items():
            _, axs = plt.subplots(1, 2, figsize=(12, 10))
            des = {k: v for k, v in sorted(val.items(),
                                           key=lambda item: item[1],
                                           reverse=True)}
            keys = [i.lower() for i in des.keys()]
            plot_pie(axs[0], list(des.values()))
            plot_bar(axs[1], keys, list(des.values()))
            plt.suptitle(key.lower() + " class distribution")
            plt.tight_layout()
            plt.savefig("analyze/" + key.lower() + ".png")
            df = pd.DataFrame([list(des.values())], columns=des.keys())
            df.to_csv("analyze/" + key + ".csv", index=False)
        plt.show()

    except AssertionError as err:
        print(f"AssertionError: {err}")
    except Exception as err:
        print(f"Error exception {err} \nline: {err.__traceback__.tb_lineno}")


if __name__ == "__main__":
    main()
