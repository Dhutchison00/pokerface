# Preflop HUD

Watches a poker table on your screen, reads it with OCR, and shows a GTO-Wizard-style
13×13 preflop chart for the player in the **bottom-centre seat** on a local web page.
The chart updates as the hand moves on.

Built for your own bots table (laid out like the ClubWPT Gold reference screenshot in
`samples/`). Use it on your own app only: real-money and sweepstakes sites ban real-time
assistance tools.

## Quick start

1. Install Python 3.11. On Windows, double-click `start.bat`. On macOS, open Terminal in the
   project folder and run `./start.sh`. The first run creates `.venv` and installs the packages.
   The macOS launcher uses CPU OCR; it requires `python3.11` on your PATH. If `./start.sh` is
   not executable, run `chmod +x start.sh` first. You can pass options to it, for example
   `./start.sh --port 9000`. If `uv` is available, the launcher uses it to create the
   environment and install dependencies; otherwise it uses Python's `venv` and pip.

   Or on Windows, run these yourself:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python -m pip install -r requirements.txt
   .\.venv\Scripts\python -m preflop_hud
   ```
   On macOS, allow Terminal (or the app launching the script) access under System Settings >
   Privacy & Security > Screen & System Audio Recording when prompted. The region picker also
   requires a Python installation with Tkinter. For Homebrew Python 3.11, install it with
   `brew install python-tk@3.11`.
2. The page opens at <http://127.0.0.1:8765/>.
3. Click **Select region**. The screen freezes and dims. Drag a box around the **whole table**
   (all seats, the blinds header and the action buttons), then release. Press Esc to cancel.
   The region is saved in `config.json`.
4. Play. The capture runs every 0.5 s, and a full read of the table takes about 0.2 s on an NVIDIA
   GPU or 0.7–0.9 s on the CPU once OCR has warmed up (start-up takes about 2–3 s). See GPU OCR below. A new reading is applied once the screen
   holds still for one frame, or once two reads agree. The chart usually updates 1–2 s after an
   action, and the program skips OCR entirely while nothing on screen changes.

Other commands:

| Command | What it does |
|---|---|
| `python -m preflop_hud --image samples\clubwpt_cutoff.webp` | Run the page against a screenshot (testing) |
| `python -m preflop_hud --check FILE` | Parse one screenshot and print the spot and chart |
| `python -m preflop_hud --select-region` | Pick the region without the web page |
| `python -m preflop_hud --port 9000 --interval 1` | Change the port or capture speed |
| `python tests\test_screenshot.py` | Regression test on the reference screenshot plus simulated spots |

**Show capture** on the page displays the last frame with every OCR box labelled with what it was
read as (stack, bet, label, name, and so on). Use it first whenever a number looks wrong.

## What it reads

| Item | How |
|---|---|
| Blinds and ante | the `200/400(50)` text under **BLINDS** |
| Stacks and bets | the `19.7 BB` text, assigned to the nearest seat or bet anchor. Chip amounts are converted with the big blind |
| Pot | `POT: 2.5 BB` (used to work out dead money and antes) |
| Seat state | `FOLD`, `SB`, `BB`, `SITTING OUT`, `ALL IN` labels, plus card-back detection |
| Button | the SB/BB labels. Falls back to the white "D" disc |
| Your cards | ranks read by OCR from an upscaled crop. Suited or offsuit comes from the colour of the rank glyph. In a four-colour deck (detected from the first blue or green card, or set `"four_color_deck": true` under `"layout"`) the same colour means suited. Otherwise red/red and black/black are told apart by the shape of the pip |
| Whose turn | the countdown number |

Positions (UTG … BTN, SB, BB) are derived for 2–10 players. Players still to act before you are
**assumed to fold**, so your chart is ready before the action reaches you. A sitting-out player
counts as a fold.

### Calibrating for your app

Seat anchors are stored **relative to the green felt's bounding box**, not the screen. Window size
and the exact box you drag don't matter, as long as the whole table is inside it. If your app's
layout differs, copy `DEFAULT_LAYOUT` from `preflop_hud/layout.py` into `config.json` under
`"layout"` and adjust it. Use **Show capture** to see the anchors (numbered dots; blue for stacks,
orange for bets).

Things the reader relies on (worth telling whoever builds the app):

- A green felt (HSV hue 40–90; change `felt_hsv_lo` and `felt_hsv_hi` otherwise)
- The blinds as `SB/BB(ante)` text near a label containing "BLIND"
- Stacks and bets as text (`12.5 BB` or chip counts). `SB`, `BB` and `FOLD` labels under the seat
- Face-up hero cards with the rank in the top-left corner, coloured by suit, with the pip below it
- 8 seats, clockwise from the hero

## The ranges

Charts come from three sources, in this order of priority. The page badges each one.

1. **Your range files** (`ranges/`, badge "your ranges"). See below.
2. **The preflop solver** (badge "solver · postflop modelled").
3. **An approximate model** (badge "approximate"). It covers spots outside the solver's tree
   (multiway pots, squeezes, limps other than the SB's) and fills in while a missing solution is
   being computed.

### How the solver works (`preflop_hud/solver.py`)

It solves a full preflop game tree with **Discounted CFR**, the regret-minimisation algorithm
real poker solvers use, over all 169 starting hands at once. Results are in chip EV.

```
first in      fold / open-raise / open-jam (open-jam only at 25bb or less); SB can also limp
vs open       fold / call / 3-bet / jam
vs 3-bet      fold / call / 4-bet jam
vs 4-bet jam  fold / call
vs jam        fold / call
vs SB limp    BB: check / raise / jam  ->  SB: fold / call / jam
```

How each ending is scored:
- **Someone wins uncontested:** exact.
- **All-in showdowns:** exact, from a 169×169 equity table (10,000 Monte Carlo boards per matchup).
- **Pots that go to the flop:** these use a **modelled postflop**, because a preflop-only solver
  can't play the flop. Your share of the pot is
  `sigmoid(γ·logit(equity) + position edge + initiative edge + playability)`. Strong hands realise
  more than their equity and weak hands less (γ ≈ 1.6, growing with depth). The player in position
  and the preflop raiser get an edge, and suited or connected hands play better. All of it fades
  towards raw equity at low stack-to-pot ratios. Every parameter is in `DEFAULT_PARAMS`.

Convergence is checked by computing each player's **best-response gain**, how much they could win
by deviating. Solves stop below 0.002 bb per hand. `tests/test_solver.py` checks:
- jam/fold trees reproduce published Nash charts (heads-up 10bb: SB shoves 58%, BB calls 37%);
- across all stack depths, AA never folds, opens widen from UTG to BTN, the BB defends wider vs
  BTN than vs UTG, and neighbouring stack depths stay consistent.

Typical 8-handed output (ante 0.125bb per player):

| Stack | UTG | HJ | CO | BTN | BB defends vs UTG / BTN |
|---|---|---|---|---|---|
| 15bb | 14% | 25% | 33% | 43% | 55% / 81% |
| 25bb | 14% | 25% | 34% | 44% | 31% / 63% |
| 60bb | 15% | 27% | 37% | 55% | 27% / 65% |

**Live versus precomputed:**
- **Live, from the real stacks** (about 0.5 s): first-in push/fold below 9.5bb effective, and
  calling a first-in shove up to 20bb.
- **Precomputed grid:** everything else. It covers 2–8 players, equal stacks of
  10, 12, 15, 18, 21, 25, 30, 35, 40, 50, 60, 80 and 100bb, and an ante of 0.125bb per player.
  The nearest stack depth is used, and the page says which.
- **On demand:** a table size or ante that isn't cached is solved in the background in a few
  seconds and saved to `data/solutions/`.
- **Rebuilding the grid** (needed after changing parameters, which needs a `MODEL_VERSION` bump):
  ```powershell
  .\.venv\Scripts\python -m preflop_hud.solver --players 8 7 6 5 4 3 2 --iters 3000
  ```
- **Bet sizes the grid was solved for:** opens 2.0bb at 20bb or less, 2.2bb up to 40bb, then 2.3bb.
  SB opens 2.5–3bb, 3-bets are 3× in position and 4× out of position, BB isolation raises are
  3.5bb, and 4-bets are jams. The page warns when an open on screen is more than 25% off the
  solved size.

**How the postflop model was calibrated:** I tuned it by hand so the output lands near typical
solver frequencies from memory. It has not been checked against a reference solver, so treat the
charts as a well-founded approximation, not GTO. Known deviations: the BB defends a bit tight vs
early-position opens at 21bb and deeper (24–31% vs UTG), and the BTN continues fairly wide vs a
CO open (about 38%). To change this, adjust `DEFAULT_PARAMS` and re-solve.

**Antes:** dead money is measured from the pot on screen (pot minus bets), so per-player antes and
big-blind antes both work. A dead-money amount that isn't cached is solved on first use.

**Known limits, all simplifications common to preflop-only solvers:**
- **Not GTO Wizard:** it doesn't solve postflop, so pots that see a flop depend on the
  realisation model.
- **No squeezes or multiway pots:** once someone calls or 3-bets, everyone behind folds. A cold
  call pays a small pot tax for the players still to act, to stand in for squeeze risk.
- **Equal stacks** in the grid; real stacks are used only in the live jam/fold spots.
- **Other simplifications:** card removal against third players is ignored, results are chip EV
  (no ICM), and bet sizes are fixed.
- **Rarely reached hands:** cells for hands that almost never reach a node are greyed out, because
  their frequencies are noise. For your own hand the page then goes by the EV of each action.

### Using your own ranges (overrides)

Put `.json` files in `ranges/`. They are re-read automatically when they change, and they win over
the built-ins when the spot matches. See `ranges/CO_RFI_20bb.json.example`:

```json
{
  "spot": "RFI",          // RFI | VS_OPEN | VS_SHOVE | VS_3BET | LIMPED
  "hero": "CO",
  "villain": "UTG",       // optional, for VS_* spots
  "stack_bb": 20,         // closest stack within stack_tolerance wins
  "stack_tolerance": 3,
  "players": 8,           // optional
  "actions": { "raise": "AA,KK,AQs:0.5,A5s-A2s,77+", "allin": "66-22" }
}
```

(The `//` comments are explanation only; real JSON files can't contain them.) Range strings
accept `AA`, `AKs`, `AKo`, `AK` (both), `77+`, `A2s+`, `KTo-K8o`, `T9s-65s`, `99-55`, and
`hand:freq` (the format most range tools export). Whatever is left over is fold. Actions are
`allin`, `raise`, `call`, `check` and `fold`. A file can also hold a list of spots.

## Files

```
preflop_hud/
  reader.py        OCR + detectors -> TableState (seats, stacks, bets, labels, hero cards)
  layout.py        felt-relative seat/bet anchors (calibration)
  spot.py          positions, action so far, effective stack -> decision point
  solver.py        preflop CFR solver, game tree, postflop realisation model, solution cache
  solved.py        maps a detected spot to a solver node (or a live jam/fold solve)
  strategy.py      chart dispatch + the approximate fallback model
  overrides.py     your range files
  equity.py        169x169 preflop equity table (Monte Carlo, cached in data/)
  evaluator.py     vectorised 7-card hand evaluator
  capture.py       screen grab (DPI-aware, multi-monitor)
  region_select.py drag-to-select overlay
  server.py        capture loop + local HTTP server
  web/index.html   the chart page
```

## GPU OCR

OCR runs on the GPU when the installed onnxruntime build can use one (CUDA first, then DirectML),
and on the CPU otherwise. The console prints `OCR device: …` at start-up and the page shows it next
to the read time. A full table read takes about 0.2 s on an RTX 5070 Ti against 0.8–1 s on the CPU.

- `start.bat` sets this up on first run when an NVIDIA driver is present. For an existing `.venv`,
  run `setup_gpu.bat` once. It swaps the CPU `onnxruntime` for `onnxruntime-gpu[cuda,cudnn]`
  (CUDA 13 + cuDNN 9 from pip, about 1.5 GB, no separate CUDA install). It needs an R580 or newer
  driver and puts the CPU build back if anything fails.
- `pip install -r requirements.txt` reinstalls the CPU `onnxruntime` over it (rapidocr depends on
  it), so run `setup_gpu.bat` again after that.
- Force a device with the `HUD_OCR_DEVICE` environment variable: `cpu`, `cuda`, `dml` or `auto` (default).
- Lines like `No registered plugin EP device found for 'CUDAExecutionProvider'` at start-up are
  onnxruntime noise. Trust the `OCR device:` line.

## Troubleshooting

- **`DLL load failed ... filename or extension is too long`** when importing onnxruntime: the
  folder path is too deep for Windows. Move the project somewhere short (such as `C:\PreflopHUD`),
  or create the venv at a short path and run it from there.
- **"Table felt not found"**: the region doesn't contain the table, or the felt isn't green. Adjust `felt_hsv_*`.
- **Wrong seat for a number**: open Show capture, then nudge that seat's `stack` or `bet` anchor in `config.json`.
- **Region is offset on a scaled display**: the capture and the picker are both per-monitor DPI-aware.
  If it still drifts, re-select the region after changing display scaling.
- `data/equity169.npy` is generated on first run if missing (takes a few minutes).
