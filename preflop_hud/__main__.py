"""python -m preflop_hud [--image FILE] [--port 8765] [--select-region] [--check FILE]"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(prog="preflop_hud", description="Live preflop chart from a poker table on screen")
    ap.add_argument("--image", help="read this screenshot instead of the screen (testing)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--interval", type=float, help="seconds between captures (default 0.5)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--select-region", action="store_true", help="pick the capture region and exit")
    ap.add_argument("--check", metavar="FILE", help="parse one screenshot, print the result and exit")
    a = ap.parse_args()

    if a.select_region:
        from .region_select import select_region
        from .server import load_config, save_config
        r = select_region()
        if r:
            cfg = load_config()
            cfg["region"] = r
            save_config(cfg)
            print("Saved region", r)
        else:
            print("Cancelled")
        return

    if a.check:
        import cv2
        from .reader import read_table
        from .spot import derive_spot
        from .strategy import Engine
        from .cards import format_range
        import numpy as np
        st = read_table(cv2.imread(a.check))
        spot = derive_spot(st)
        print(json.dumps({k: v for k, v in spot.items() if k not in ("positions",)}, indent=1, default=str))
        print("hero cards:", st.hero_cards)
        ch = Engine().solve(spot)
        print(ch.get("title"), "|", ch.get("method"), ch.get("pct"))
        if ch.get("freqs"):
            F = np.array(ch["freqs"])
            for i, act in enumerate(ch["actions"]):
                if act not in ("fold", "check"):
                    print(f"  {act}: {format_range(F[:, i])}")
        return

    from .server import serve
    serve(port=a.port, image=a.image, open_browser=not a.no_browser, interval=a.interval)


if __name__ == "__main__":
    sys.exit(main())
