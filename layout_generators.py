import klayout.db as db
import numpy as np

def _layout(dbu=0.001):
    # 1 DBU = 1 nm when dbu=0.001 (µm units)
    L = db.Layout()
    L.dbu = dbu
    return L


def finfet_layout(filename="finfet.gds", num_fins=8, num_gates=2):
    L = _layout()
    top = L.create_cell("FINFET")
    fin_l  = L.layer(db.LayerInfo(1, 0))  # Active / Fin
    gate_l = L.layer(db.LayerInfo(2, 0))  # Poly / Gate

    FIN_W, FIN_H, FIN_PITCH = 7, 100, 28   # nm → DBU
    GATE_W, GATE_SP = 12, 40

    for i in range(num_fins):
        x = i * FIN_PITCH
        top.shapes(fin_l).insert(db.Box(x, 0, x + FIN_W, FIN_H))

    cell_w = num_fins * FIN_PITCH
    for g in range(num_gates):
        gx = g * (GATE_W + GATE_SP) + GATE_SP
        top.shapes(gate_l).insert(
            db.Box(-10, gx, cell_w + 10, gx + GATE_W)
        )

    L.write(filename)


def dram_layout(filename="dram.gds", rows=16, cols=16):
    L = _layout()
    top = L.create_cell("DRAM")
    node_l = L.layer(db.LayerInfo(1, 0))  # Capacitor nodes
    wl_l   = L.layer(db.LayerInfo(2, 0))  # Word lines
    bl_l   = L.layer(db.LayerInfo(3, 0))  # Bit lines

    CELL_X, CELL_Y = 38, 38   # cell pitch (nm)
    NODE_W, NODE_H = 20, 20   # capacitor pad
    WL_W, BL_W = 10, 10

    for r in range(rows):
        for c in range(cols):
            ox, oy = c * CELL_X, r * CELL_Y
            top.shapes(node_l).insert(
                db.Box(ox + 9, oy + 9, ox + 9 + NODE_W, oy + 9 + NODE_H)
            )
        y = r * CELL_Y
        top.shapes(wl_l).insert(db.Box(0, y, cols * CELL_X, y + WL_W))

    for c in range(cols):
        x = c * CELL_X
        top.shapes(bl_l).insert(db.Box(x, 0, x + BL_W, rows * CELL_Y))

    L.write(filename)


def sram_cell_layout(filename="sram.gds", tile_x=4, tile_y=4):
    """6T SRAM bit cell array (simplified geometry)."""
    L = _layout()
    top = L.create_cell("SRAM")
    active_l = L.layer(db.LayerInfo(1, 0))
    poly_l   = L.layer(db.LayerInfo(2, 0))
    metal_l  = L.layer(db.LayerInfo(3, 0))

    # Single 6T cell template (200nm × 400nm)
    CW, CH = 200, 400

    def draw_cell(ox, oy):
        top.shapes(active_l).insert(db.Box(ox+20, oy+10,  ox+55,  oy+110))
        top.shapes(active_l).insert(db.Box(ox+145, oy+10, ox+180, oy+110))
        top.shapes(active_l).insert(db.Box(ox+20,  oy+290, ox+55,  oy+390))
        top.shapes(active_l).insert(db.Box(ox+145, oy+290, ox+180, oy+390))
        top.shapes(active_l).insert(db.Box(ox+70,  oy+10,  ox+130, oy+60))
        top.shapes(active_l).insert(db.Box(ox+70,  oy+340, ox+130, oy+390))
        top.shapes(poly_l).insert(db.Box(ox+60,  oy+50,  ox+75,  oy+350))
        top.shapes(poly_l).insert(db.Box(ox+125, oy+50,  ox+140, oy+350))
        top.shapes(metal_l).insert(db.Box(ox+55,  oy+190, ox+145, oy+210))

    for row in range(tile_y):
        for col in range(tile_x):
            draw_cell(col * CW, row * CH)

    L.write(filename)


def nand_flash_layout(filename="nand.gds", strings=12, cells_per_string=16):
    """NAND string column array."""
    L = _layout()
    top = L.create_cell("NAND_FLASH")
    tun_l   = L.layer(db.LayerInfo(1, 0))  # Tunnel oxide / channel
    fgate_l = L.layer(db.LayerInfo(2, 0))  # Floating gates
    wl_l    = L.layer(db.LayerInfo(3, 0))  # Word lines (control gates)

    STR_PITCH = 50   # string pitch nm
    CELL_H    = 25   # cell height nm
    FG_W      = 20   # floating gate width

    for s in range(strings):
        sx = s * STR_PITCH
        # Channel column
        top.shapes(tun_l).insert(
            db.Box(sx+5, 0, sx+25, cells_per_string * CELL_H)
        )
        for c in range(cells_per_string):
            cy = c * CELL_H
            top.shapes(fgate_l).insert(db.Box(sx+5, cy+5, sx+5+FG_W, cy+18))

    for c in range(cells_per_string):
        cy = c * CELL_H + 5
        top.shapes(wl_l).insert(
            db.Box(0, cy, strings * STR_PITCH, cy + 5)
        )

    L.write(filename)
