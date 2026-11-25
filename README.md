# Preflop HUD

Watches a poker table on your screen, reads it with OCR, and shows a GTO-Wizard-style
13×13 preflop chart for the player in the **bottom-centre seat** on a local web page.
The chart updates as the hand moves on.

Built for your own bots table (laid out like the ClubWPT Gold reference screenshot in
`samples/`). Use it on your own app only: real-money and sweepstakes sites ban real-time
assistance tools.

## Quick start

1. Install Python 3.11, then double-click `start.bat`. The first run creates `.venv` and installs
   the packages. Or run these yourself:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python -m pip install -r requirements.txt
   .\.venv\Scripts\python -m preflop_hud
   ```
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
| Board | white card faces in the middle of the felt. Ranks by OCR (with a single-glyph retry), suit from the rank's colour: red hearts, blue diamonds, green clubs, black spades. Shown in the Spot panel; display only for now |
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

- A green or red felt (green: HSV hue 40–90 in `felt_hsv_lo`/`felt_hsv_hi`; other themes in `felt_hsv_extra`)
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
  tracker.py       rebuilds the hand's action from successive reads
  postflop.py      live heads-up advice, flop to river: each street solved by GTOpen from its start
  multiway.py      3-way pots: multiway preflop ranges, equity, the lean 3-way turn/river spots
  gtopen.py        GTOpen preflop solves, exports and the chart store
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

## Hand tracker and recording

The Spot panel lists each street's action as the hand plays out (`BTN raise 2.5, SB fold, BB call 2.5 |
BB check, BTN bet 1.8`). One frame doesn't show earlier action, so the tracker compares successive
reads: bet chips growing are calls, bets or raises; FOLD and CHECK labels (or a seat's cards
disappearing) are folds and checks; and a new board card closes the street. Bots can act faster
than the screen is read, so the HUD reads every 0.25 s while you're in a hand, puts several actions
seen in one frame in seat order, and marks with `*` the actions it only deduced when the street
closed (a call swept into the pot together with the next card).

**Record** (top of the page) saves every read, as the frame PNG plus the tracker's view, to
`recordings/<time>/` until you click it again. **Advice is off while recording**: no chart, no
hand advice and no solving, only reading, tracking and saving, so recording can run on other
tables without giving advice. It comes back when you stop. Replay a recording through a fresh tracker with:

```powershell
.\.venv\Scripts\python -m preflop_hud.tracker replay recordings\<time>
```

Add `--reread` to read the saved frames again (after a reader change). Recordings are git-ignored;
they're the test data for the tracker and the postflop advice.

## Tournament or cash

The **Tournament / Cash** switch at the top of the page picks the game type. It's saved in
`config.json` under `"game"` and re-solves the current spot at once.

| Mode | What changes |
|---|---|
| Tournament · Chip EV | The default: antes read from the pot, every chip worth the same |
| Tournament · ICM stage | Final table, two tables, 10% left, bubble or half the field (payouts of a 1,000-player event, 15% paid). Only GTOpen charts solved for that stage are ICM-aware. The HUD's own solver has no ICM, so other spots show chip EV with a note |
| Cash | Rake taken from pots that see a flop (`"rake_pct"` 5, `"rake_cap"` 3 bb by default), 2.5 bb opens and 3 bb from the SB. Usually no antes |

Each mode has its own solver cache (`data/solutions/v2_cash_r5c3_…` for cash) and its own GTOpen
exports, so switching never mixes charts. To precompute cash charts with the HUD's solver, run
`python -m preflop_hud.solver --mode cash --players 6 --stacks 100`.

## GTOpen preflop charts (optional)

[GTOpen](https://github.com/MatthewPDingle/GTOpen) solves the full multiway preflop tree (squeezes,
sized 3-bets, blockers) on an NVIDIA GPU, with flop values learned from exact flop solves. That
takes minutes per grid point, or hours with limps, so it runs offline. The HUD then reads the
exported charts from `data/gtopen/` and shows them with a **GTOpen** badge. Anything not exported
falls back to the HUD's own solver. GTOpen charts have no per-hand EVs.

Build GTOpen once (needs Rust and the Visual Studio C++ build tools), then point the HUD at it with
`--gtopen-dir`, the `HUD_GTOPEN_DIR` environment variable or `"gtopen_dir"` in `config.json`:

Clone to a short path: the repository's research files exceed Windows' 260-character limit otherwise.

```powershell
git -c core.longpaths=true clone https://github.com/MatthewPDingle/GTOpen.git C:\GTOpen; cd C:\GTOpen
python -m pip install --target .cuda-nvrtc nvidia-cuda-nvrtc-cu12
cargo build --release -p server --features gpu
```

| Command | What it does |
|---|---|
| `python -m preflop_hud.gtopen solve --players 8 --stack 60 --limp` | Solve one grid point, saving a checkpoint every 100 iterations; rerun the same command to resume. Exports when done |
| `python -m preflop_hud.gtopen export --players 8 --stack 60 --limp` | Export a finished save to `data/gtopen/` |
| `python -m preflop_hud.gtopen compare --players 8 --stack 60 --limp` | Print every exported spot next to the HUD solver's |
| `python -m preflop_hud.gtopen batch --players 8 --stack 25 21 18 15 30 --limp` | Solve and export several depths in turn; with `--limp`, a depth whose limp tree GTOpen refuses (or can't keep on the GPU) is solved without limps. Skips depths already exported, keeps the PC from idle-sleeping while it runs, and deletes each 2–4 GB checkpoint once exported (`--keep-saves` to keep them) |

- `--stack` is one of the HUD's grid depths. Antes default to 0.125 bb per player in tournaments and
  0 in cash (`--ante-per-player`). A chart is used when the table size, grid stack and game type match
  and the antes are within 0.5 bb.
- Game type: `--mode cash` (with `--rake-pct` / `--rake-cap`) or `--icm-stage ft|t2|pct10|bubble|pct50`.
  ICM games use GTOpen's plain "balanced" flop values without blockers: its learned values and blocker
  model don't support ICM.
- **Limps.** GTOpen can only allow limping for every seat, not just the small blind as the HUD's tree
  does. Without `--limp`, the small blind's first-in and the lines it opens stay with the HUD's
  solver. With `--limp` those come from GTOpen too, but the tree is roughly 7–15× larger (8-handed
  60 bb: 1.65M nodes, about 6 s per iteration on an RTX 5070 Ti).
- The checkpoints and logs live in GTOpen's `saves/preflop/` folder.

## Live postflop advice

From the flop to the river, while you're heads-up (neither player all-in), the left panel switches
from the preflop chart to advice for your exact hand: how often to take each action, how much each
action loses against the best one (EV loss, in bb), your equity, and how the whole range plays the
spot. It needs GTOpen built (see the section below), with `"gtopen_dir"` in `config.json`.

1. **Flop ranges** come from the preflop line the tracker saw: each player's range is their own reach
   at their last preflop decision times how often they took the action they took, from the same
   charts the HUD shows preflop (GTOpen exports first, else the HUD solver's cache). Lines covered:
   open + call, open + 3-bet + call, SB limp + BB check, SB limp + BB raise + SB call, and a limp from
   any other seat that's checked by the BB or iso-raised and called heads-up. The charts almost never
   limp outside the SB (0.2% of hands or less at 10–35bb, up to ~3% at 100bb), so the solver's nodes
   after such a limp aren't converged: the limper's range is assumed to be its open-raise range, an
   iso-raise the raiser's open range (the BB's: its range against an SB limp), noted on the panel.
2. **Turn and river ranges** carry the betting so far: each street is solved on its own, from its
   start, and each player's range for the next street is their reach at the end of the previous
   street's solved tree, combo by combo. Re-solving from each street's start gives the same answers
   as the full flop tree (checked on a recorded turn: your hand 99.7% vs 99.4% check, the range 35%
   vs 37%), at the real pot and stacks. A player who takes a line the solver almost never takes
   (under 2% of their range) keeps the range they had before it, noted; so does your hand if it's
   nearly gone from your range.
3. **The spot** is the board so far, plus the pot and stacks the street started with (medians over
   several reads, taken before that street's bets go in). Flop trees bet 33% or 75% of the pot
   (75% on later streets), raise 60% (of the pot after calling), and treat anything within 85% of
   the stack as all-in. Deep flops (stacks 4.5× the pot or more, or a tree over 6 GB of VRAM) use a
   smaller menu, one 33% flop bet and all-in turn/river raises, to answer in seconds (noted). Turn-
   and river-rooted trees are tiny (thousands of nodes), so they bet 33% or 75% (and all-in on the
   river), raise 60% or all-in, allow four raises, and let the player out of position lead into the
   last street's bettor. Cash mode applies the rake; tournaments are chip EV postflop for now (noted
   in ICM mode).
4. **Real bet sizes**: a bet or raise that isn't in the menu (more than 15% from its nearest size, like
   the bots' 12% and 25% bets) is added to that street's menu and the street solved again, noted. The
   first tree's answer, with the bet mapped to its nearest size, shows meanwhile, marked as
   re-solving. Turn and river re-solves take a fraction of a second; a flop one, a couple.
5. **Solving** starts the moment a street's first card shows, on a gto-server of the HUD's own (port
   3741, started while preflop is played). In the HUD loop on recorded hands, on an RTX 5070 Ti: a
   flop answer about 2 s after the flop in a raised pot (6 s in a deep limped pot, first at 2.5% of
   the pot from equilibrium, refined to 0.3% while you look, "refining…"); turn and river answers
   0.5–1.5 s after their card. The panel shows "Solving the turn…" (etc.) until then, and the
   iterations, accuracy and GPU/CPU underneath.
6. **Your decision**: the HUD follows the street's logged actions down its solved tree to your
   decision. If your hand is almost never in your range on that line, it's kept in at a small weight
   and noted.

### 3-way and bigger pots

GTOpen's three-player solver (beta) is used where it answers live, and your equity everywhere else.

- **Ranges**: the charts have no multiway nodes, so each player's range comes from their own heads-up
  node against the first raiser: the opener's open, a caller's call against that open (an overcall is
  its call against a lone open), limpers and iso-raisers as in heads-up limped pots. All noted. 3-bet
  multiway pots aren't covered.
- **The flop**: your equity against every player's range (Monte Carlo, card removal between all hands,
  seeded so it holds still), and the pot odds when you face a bet. A 3-way flop is far too big to solve
  live on one GPU: with one bet size, 2.7M nodes needing 14 GB, still 4.6% of the pot from equilibrium
  after ten minutes.
- **The turn**, when the flop was checked through by everyone: solved 3-way from the turn's start, with
  one bet size (75%) and all-in raises (75% and all-in on the river inside it), a first answer in about
  3 s at 18bb. The flop's checking ranges are taken as the preflop ranges, which is rough: on a
  heads-up test turn, skipping a checked flop's narrowing made the range bet about half as often as it
  should (your own hand's advice there was the same). Noted on the panel.
- **The river** after such a turn: each player's range is read from the turn's solved tree (where they
  first act on the river). Still 3-way: solved with 33%, 75% and all-in bets, in about a second. A
  player folded the turn: solved heads-up, from the same ranges.
- **A reduced-tree range estimator** for 3-way flops is built but off (`ESTIMATOR_LIVE` in
  `multiway.py`): a 3-way flop with one bet size and the turn and river checked down solves in ~3 s,
  but heads-up, where the full flop tree is solvable, that reduction gave ranges further from the full
  tree's than no narrowing at all (checking ranges come out capped when the flop is the only street to
  bet a strong hand), and one turn bet size didn't fix it. `python -m preflop_hud.validate_multiway`
  checks it against full 3-way flop solves on synthetic hands (hours of GPU, and ~8 GB of free RAM per
  reference tree: run it overnight with other programs closed).
- **Everything else** shows your equity with the reason: a flop that was bet, side pots (a player
  all-in), 4-way pots. Equity against ranges that the betting since hasn't narrowed is likely higher
  than your real equity, and the panel says so; it isn't a call-or-fold verdict (more bets can follow).
  3-way advice gives each action's frequency but no EV loss (the 3-way solver doesn't report per-action
  EVs).

Shown instead of advice, with the reason: preflop lines not covered above (over-limps, limp-reraises,
4-bets, cold calls), hands where tracking started after the preflop action, pots where you're all-in
or nobody with chips is left to bet against you, and an unread hole-card suit. Advice is off while
recording, as everywhere.

## Troubleshooting

- **`DLL load failed ... filename or extension is too long`** when importing onnxruntime: the
  folder path is too deep for Windows. Move the project somewhere short (such as `C:\PreflopHUD`),
  or create the venv at a short path and run it from there.
- **"Table felt not found"**: the region doesn't contain the table, or the felt is neither green nor red. Add its HSV range to `felt_hsv_extra`.
- **Wrong seat for a number**: open Show capture, then nudge that seat's `stack` or `bet` anchor in `config.json`.
- **Region is offset on a scaled display**: the capture and the picker are both per-monitor DPI-aware.
  If it still drifts, re-select the region after changing display scaling.
- `data/equity169.npy` is generated on first run if missing (takes a few minutes).
