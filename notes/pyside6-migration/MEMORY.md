# Memory use over long sessions (v1.6.0 candidate)

Measured offscreen on Linux (x86-64, 8 GB), with `tools/gui_stress.py` and a probe that separates the steps of
the stress loop. RSS is the process's resident memory; "after trim" is RSS after `malloc_trim(0)` (memory the
allocator could return), so growth that survives the trim is held, not fragmentation.

## Results

| Loop | Binding | Rounds | Crashes | RSS first -> last | Per round | Widgets | Qt wrappers |
|---|---|---|---|---|---|---|---|
| default pair, pages + views | PySide6 6.12 / Py 3.13 | 400 | 0 | 280 -> 535 MB | 0.64 MB | 2017 constant | - |
| default pair, pages + views | PyQt6 6.11 / Py 3.13 | 400 | 0 | 239 -> 415 MB | 0.44 MB | 2017 constant | - |
| all examples, pages + views | PySide6 6.12 / Py 3.13 | 100 | 0 | 300 -> 542 MB | 2.4 MB | 1978 constant | - |
| all examples, pages + views | PyQt6 6.11 / Py 3.13 | 72 (time limit) | 0 | 255 -> 399 MB | 2.0 MB | 1978 constant | - |
| default pair, pages + views | PySide6 6.11.3 / Py 3.11 | 150 | 0 | 279 -> 379 MB | 0.67 MB | 2017 constant | - |
| pages only (40 rounds) | PySide6 / PyQt6 | 40 | 0 | flat / flat | 0 | constant | 2785 constant |
| geometry views only | PySide6 / PyQt6 | 40 | 0 | flat / flat | 0 | constant | 2777 constant |
| example loads only, before the registry fix | PySide6 / PyQt6 | 40 | 0 | +1.0 / +0.8 MB per round | | constant | 2777 constant |
| example loads only, candidate (registry fix) | PySide6 6.12 | 400 | 0 | 275 -> 450 MB (after trim 252 -> 441) | 0.44 MB | 2013 constant | 2777 constant |

The growth is linear over 400 rounds; it does not level off within that range. It comes only from loading
examples, with both bindings, and no widget or Qt wrapper accumulates.

## Retained allocations identified

1. **Fixed: Display Settings registry.** Every example load rebuilds the condition rows; their entries (about 60
   per load, each with a style lambda) stayed in `MainWindow._styled_widgets` until the next Display Settings change.
   Now pruned when the list doubles (`_prune_styled_widgets`), test
   `test_example_loads_do_not_accumulate_styled_widget_entries`. Python objects added per load: ~280 -> ~110.
   The defect is also in the PyQt6 code (`audit/v1.6.0`), not ported there.
2. **Open: equation previews (matplotlib).** `pinnstudio/ui/equation.py:render` draws each typeset equation with a
   new matplotlib Figure and Agg canvas. Over 400 loads the GUI keeps 6 matplotlib mathtext font sets per load
   (`StixSansFonts` with its `FontProperties` and an `lru_cache`, +2340 each), and on its own `render()` grows RSS
   by about 20 KB per call (3000 calls: 144 -> 202 MB, standalone). heaptrack attributes about 1.7 MB per 8 rounds
   to `QImage` data of label pixmaps set from the preview's timer, and about 0.9 MB to FreeType. Follow-up: reuse one
   figure/canvas for the previews, or cache rendered pixmaps by (latex, colour), and recheck.
3. **Unattributed: Python allocator, about 50 % of the rest** (heaptrack shows it only as the interpreter's own
   allocations).

## Check of item 2 (2026-11-03)

Caching each drawn equation (bytes of the image, keyed by equation, colour and size, so that no matplotlib figure
is drawn again) does **not** remove the growth: 3000 `render()` calls of 6 equations in 2 colours grew the
resident memory from 127 to 202 MB with the cache and without it, and 9000 calls from 127 to 332 / 333 MB
(PySide6 6.11.3, Python 3.11, offscreen; current RSS from `/proc/self/statm`). So the retained memory comes from
creating the `QImage`/`QPixmap` each time, not from matplotlib redrawing; the cache was not kept. A fix would
have to reuse the pixmap objects themselves, which ties Qt objects to a module-level cache (a risk at shutdown);
left as a known issue.

## Size in practice

About 0.4-0.6 MB per example switch: some 0.5 GB after 1000 switches in one session. Closing and reopening
PINNStudio returns it. Not a release blocker; listed as a known issue.
