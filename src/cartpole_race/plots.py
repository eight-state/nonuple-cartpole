"""Capture-basin plots: 2D physical slices + modal-coordinate anisotropy.

Per the proposal M1 build list ('2D + modal-coordinate plots showing
anisotropy'). Consumes the parquet produced by :mod:`mapper`. Uses a
non-interactive matplotlib backend so plotting works headless under workers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_basin_2d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: str | Path,
    title: str = "Capture basin",
) -> Path:
    """Scatter success/failure over two physical state components.

    Args:
        df: Mapper results with ``success`` plus the named columns.
        x_col: Column for the horizontal axis.
        y_col: Column for the vertical axis.
        out_path: PNG output path.
        title: Plot title.

    Returns:
        The written path.
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ok = df["success"] == True  # noqa: E712
    ax.scatter(
        df.loc[~ok, x_col], df.loc[~ok, y_col],
        s=6, c="#cc3333", alpha=0.5, label="fail",
    )
    ax.scatter(
        df.loc[ok, x_col], df.loc[ok, y_col],
        s=6, c="#2a7", alpha=0.6, label="success",
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_modal_basin(
    df: pd.DataFrame,
    z_cols: list[str],
    out_path: str | Path,
    title: str = "Modal-coordinate basin (anisotropy)",
) -> Path:
    """Scatter success/failure over the two leading unstable modal coords.

    Demonstrates anisotropy: the basin is an ellipse, not a circle, in modal
    coordinates ``z_u``. If fewer than two modal columns exist, the second axis
    falls back to the modal norm.

    Args:
        df: Mapper results carrying modal columns (e.g. ``z0, z1, ...``).
        z_cols: Modal-coordinate column names (>=1).
        out_path: PNG output path.
        title: Plot title.

    Returns:
        The written path.
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ok = df["success"] == True  # noqa: E712
    xz = df[z_cols[0]]
    if len(z_cols) >= 2:
        yz = df[z_cols[1]]
        ylabel = z_cols[1]
    else:
        yz = np.linalg.norm(df[z_cols].to_numpy(), axis=1)
        ylabel = "||z_u||"
    ax.scatter(xz[~ok], yz[~ok], s=6, c="#cc3333", alpha=0.5, label="fail")
    ax.scatter(xz[ok], yz[ok], s=6, c="#2a7", alpha=0.6, label="success")
    ax.set_xlabel(z_cols[0])
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def make_all_plots(parquet_path: str | Path, out_dir: str | Path) -> list[Path]:
    """Generate the standard basin plot set from a mapper parquet.

    Args:
        parquet_path: Path to a mapper ``*.parquet``.
        out_dir: Directory for PNGs (created if missing).

    Returns:
        List of written PNG paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet_path)
    written: list[Path] = []

    # Physical slices.
    written.append(
        plot_basin_2d(df, "theta0", "thetad0", out_dir / "basin_theta0.png",
                      "Basin: theta_1 vs thetadot_1")
    )
    written.append(
        plot_basin_2d(df, "x_cart", "xdot", out_dir / "basin_cart.png",
                      "Basin: x_cart vs xdot")
    )

    # Modal slice (if modal columns present).
    z_cols = [c for c in df.columns if c.startswith("z")]
    if z_cols:
        written.append(
            plot_modal_basin(df, z_cols, out_dir / "basin_modal.png")
        )
    return written
