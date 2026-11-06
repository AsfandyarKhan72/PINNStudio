# Burgers 1D demo recording

`pinnstudio_burgers_readme.gif` is the animation at the top of the README: 960 × 552, 10 fps, 46.5 s, 2.4 MB. It is a screen recording of PINNStudio 1.6.0 solving the built-in example *Burgers 1D – Raissi et al.* with its *Default* settings. The window, plots, Training Log and numbers come from a real run. Nothing was redrawn.

## The run

- **Machine:** Linux, NVIDIA GeForce RTX 4090 (driver 550.144.03).
- **Software:** Python 3.11.5, PINNStudio 1.6.0 (installed from the release wheel), PyTorch 2.14.1+cu126, PySide6 6.11.3, DeepXDE 1.15.1.
- **Settings:** the example's *Default* settings, unchanged (Raissi et al.'s setup):
  - 8 × 20 tanh network;
  - one L-BFGS phase (recommended settings, up to 50,000 iterations);
  - float32;
  - random seed 2026.
- **Training time:** 39 s ("Finished · in 0:39" in the status bar). L-BFGS stopped on its own at iteration 4,193.
- **Final loss:** 3.258 × 10⁻⁶.
- **Relative L₂ error against the reference solution:**
  - 7.44, 5.34, 3.85, 6.49 and 3.94 × 10⁻⁴ at t = 0, 0.25, 0.5, 0.75 and 1;
  - mean 5.41 × 10⁻⁴; all reference times together 5.79 × 10⁻⁴.
  - From the run's `error_analysis/error_metrics.txt` and `error_spacetime.txt`.
  - A second recording gave the same errors and the same stopping iteration.

## How it was recorded and edited

- **Display:** PINNStudio ran on a virtual X display (Xvfb) of 1650 × 950 pixels, its default window size. There is no window manager, so no title bar.
- **Input:** the mouse movements and clicks are real X11 input events (XTEST), sent by a short script.
- **Capture:** `ffmpeg -f x11grab -framerate 30`, in one continuous take of 1 min 41 s.
- **Editing:**
  - The pauses between steps were cut.
  - The training part, from just after the Solve click until the run finished, is shown at **2× speed**; everything else plays at real speed.
  - There are no overlays.
  - The GIF was made with ffmpeg (`palettegen`/`paletteuse`, 256 colours, 960 px wide, 10 fps).

The full take and a real-time cut (MP4) are kept outside the repository.
