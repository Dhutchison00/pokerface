"""Table layout calibration.

Every anchor is expressed relative to the bounding box of the green felt:
    u = (x - felt_x) / felt_w      v = (y - felt_y) / felt_h
so the layout survives window resizing and whatever rectangle you drag as the
capture region (as long as the whole table is inside it).

Seats are listed CLOCKWISE starting from the hero (bottom centre), which is
the direction the action moves in.  Measured from the ClubWPT Gold reference
screenshot; tweak in config.json ("layout") if your app differs.
"""

DEFAULT_LAYOUT = {
    # HSV range of the felt (OpenCV H is 0..180)
    "felt_hsv_lo": [40, 60, 40],
    "felt_hsv_hi": [90, 255, 255],
    # other table themes, tried as well (the largest blob wins): red, which wraps around hue 0
    "felt_hsv_extra": [[[[0, 120, 50], [8, 255, 255]], [[172, 120, 50], [180, 255, 255]]]],
    "seats": [
        # stack: where the "19.7 BB" text sits.   bet: where that seat's bet chips land
        {"stack": [0.540, 1.100], "bet": [0.420, 0.680]},   # 0 hero, bottom centre
        {"stack": [0.095, 0.923], "bet": [0.250, 0.750]},   # 1 bottom left
        {"stack": [-0.060, 0.448], "bet": [0.170, 0.410]},  # 2 left
        {"stack": [0.062, 0.051], "bet": [0.292, 0.157]},   # 3 top left
        {"stack": [0.456, -0.087], "bet": [0.450, 0.120]},  # 4 top centre
        {"stack": [0.933, 0.051], "bet": [0.708, 0.157]},   # 5 top right
        {"stack": [1.056, 0.452], "bet": [0.830, 0.410]},   # 6 right
        {"stack": [0.900, 0.925], "bet": [0.750, 0.750]},   # 7 bottom right
    ],
    # the app's 6-seat table puts its side seats lower (beside the bottom of the felt), where no 8-seat
    # anchor reaches: each frame takes the layout that has more stacks next to its anchors (8 on a tie)
    "seats_alt": [[
        {"stack": [0.540, 1.100], "bet": [0.420, 0.680]},   # 0 hero, bottom centre
        {"stack": [0.002, 0.748], "bet": [0.200, 0.620]},   # 1 left, low
        {"stack": [0.045, 0.099], "bet": [0.292, 0.157]},   # 2 top left
        {"stack": [0.456, -0.087], "bet": [0.450, 0.120]},  # 3 top centre
        {"stack": [0.949, 0.099], "bet": [0.708, 0.157]},   # 4 top right
        {"stack": [0.993, 0.750], "bet": [0.800, 0.620]},   # 5 right, low
    ]],
    # hero hole-card area
    "hero_cards": [0.44, 0.76, 0.61, 1.00],          # u0, v0, u1, v1
    # community-card area (between the pot label and the bet chips)
    "board": [0.22, 0.27, 0.78, 0.66],
    # card-back sample box relative to each seat's stack anchor (du0, dv0, du1, dv1)
    "card_back_box": [-0.035, -0.22, 0.035, -0.13],
    "card_back_min_frac": 0.15,   # measured ~0.33 with cards, 0.0 without
    # max distance (in felt widths) for a text to belong to a seat
    "seat_radius": 0.085,
    "bet_radius": 0.14,
}
