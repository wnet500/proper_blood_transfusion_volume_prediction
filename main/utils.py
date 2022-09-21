import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

from pathlib import Path
from sklearn.metrics import r2_score
from typing import List, Union


def adjust_pred_value(x):
  """
  모델 예측값이 음수일 경우 0, 3.5보다 작으면 반올림, 3.5이상이면 올림 적용
  """
  x = np.where(x < 0, 0, x)
  x = np.where(x < 3.5, np.round(x), np.ceil(x))

  return x


def get_adjusted_r2(true_vals, predicted_vals, num_of_vals):
  """
  adjusted r2 계산
  """
  adj_r2 = 1 - (1 - r2_score(true_vals, predicted_vals)) * (len(true_vals) - 1) / (len(true_vals) - num_of_vals - 1)

  return adj_r2


def get_95_conf_interval(x):
  """
  95% 신뢰구간 계산
  """
  lower, upper = st.t.interval(
      alpha=0.95,
      df=len(x) - 1,
      loc=np.mean(x),
      scale=st.sem(x)
  )
  return [np.mean(x), lower, upper]


def disable_logging_and_userwaring():
  """
  torch lighting 모델링 warning 제거
  """
  import warnings
  import logging
  warnings.filterwarnings("ignore", category=UserWarning)
  logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def plot_blandaltman(
    x,
    y,
    agreement=1.96,
    xaxis="mean",
    confidence=0.95,
    annotate=True,
    scatter_kws=dict(color="tab:blue", alpha=0.8),
    figsize=(4.5, 4.5),
    dpi=100,
    ax=None,
):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats
    import matplotlib.pyplot as plt
    import matplotlib.transforms as transforms
    """
    The code is an adaptation of the
    `Pingouin <https://github.com/raphaelvallat/pingouin>`_ package.
    """
    # Safety check
    assert xaxis in ["mean", "x", "y"]
    # Get names before converting to NumPy array
    xname = x.name if isinstance(x, pd.Series) else "x"
    yname = y.name if isinstance(y, pd.Series) else "y"
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == 1 and y.ndim == 1
    assert x.size == y.size
    assert not np.isnan(x).any(), "Missing values in x or y are not supported."
    assert not np.isnan(y).any(), "Missing values in x or y are not supported."

    # Calculate mean, STD and SEM of x - y
    n = x.size
    dof = n - 1
    diff = x - y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    mean_diff_se = np.sqrt(std_diff**2 / n)
    # Limits of agreements
    high = mean_diff + agreement * std_diff
    low = mean_diff - agreement * std_diff
    high_low_se = np.sqrt(3 * std_diff**2 / n)

    # Define x-axis
    if xaxis == "mean":
      xval = np.vstack((x, y)).mean(0)
      xlabel = f"Mean of {xname} and {yname}"
    elif xaxis == "x":
      xval = x
      xlabel = xname
    else:
      xval = y
      xlabel = yname

    # Start the plot
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    stdev = .005 * (max(xval) - min(xval))
    xval = xval + np.random.randn(len(xval)) * stdev

    # Plot the mean diff, limits of agreement and scatter
    ax.scatter(xval, diff, **scatter_kws)
    ax.axhline(mean_diff, color="k", linestyle="-", lw=2)
    ax.axhline(high, color="k", linestyle=":", lw=1.5)
    ax.axhline(low, color="k", linestyle=":", lw=1.5)

    # Annotate values
    if annotate:
      loa_range = high - low
      offset = (loa_range / 100.0) * 1.5
      trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
      xloc = 0.98
      ax.text(xloc, mean_diff + offset, "Mean", ha="right", va="bottom", transform=trans)
      ax.text(xloc, mean_diff - offset, "%.2f" % mean_diff, ha="right", va="top", transform=trans)
      ax.text(
          xloc, high + offset, "+%.2f SD" % agreement, ha="right", va="bottom", transform=trans
      )
      ax.text(xloc, high - offset, "%.2f" % high, ha="right", va="top", transform=trans)
      ax.text(xloc, low - offset, "-%.2f SD" % agreement, ha="right", va="top", transform=trans)
      ax.text(xloc, low + offset, "%.2f" % low, ha="right", va="bottom", transform=trans)

    # Add 95% confidence intervals for mean bias and limits of agreement
    if confidence is not None:
      assert 0 < confidence < 1
      ci = dict()
      ci["mean"] = stats.t.interval(confidence, dof, loc=mean_diff, scale=mean_diff_se)
      ci["high"] = stats.t.interval(confidence, dof, loc=high, scale=high_low_se)
      ci["low"] = stats.t.interval(confidence, dof, loc=low, scale=high_low_se)
      ax.axhspan(ci["mean"][0], ci["mean"][1], facecolor="tab:grey", alpha=0.2)
      ax.axhspan(ci["high"][0], ci["high"][1], facecolor="tab:blue", alpha=0.2)
      ax.axhspan(ci["low"][0], ci["low"][1], facecolor="tab:blue", alpha=0.2)

    # Labels and title
    ax.set_ylabel(f"{xname} - {yname}")
    ax.set_xlabel(xlabel)
    sns.despine(ax=ax)
    return ax


def save_blandaltman(
    observed_values: Union[List, np.ndarray],
    predicted_values: Union[List, np.ndarray],
    file_name: str,
    xticks_step: int = 3
):
  """bland altman 플랏을 보여주고, png 파일을 output/plots 폴더에 저장합니다.

  Args:
      observed_values (Union[List, np.ndarray]): true values
      predicted_values (Union[List, np.ndarray]): model predicted values
      file_name (str): 저장할 png 파일 이름
  """
  ouput_dir = Path(__file__).parent.parent.joinpath("output")

  df = pd.DataFrame({"Observed": observed_values, "Predicted": predicted_values})

  ax = plot_blandaltman(
      df["Observed"],
      df["Predicted"],
      xaxis="x",
      dpi=300
  )
  plt.xticks(range(0, int(max(observed_values)) + 1, xticks_step))
  plt.tight_layout()
  plt.savefig(ouput_dir.joinpath("plots").joinpath(f"{file_name}.png"), dpi=300)
  plt.show()
