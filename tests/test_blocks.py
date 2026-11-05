"""The block tree of a twin's asset graph (plyxis/lang/blocks.py): found from the graph, no names of plants or lines."""
import os

import pytest

from plyxis.lang import compile_model
from plyxis.lang.blocks import find_structure, used_blocks
from plyxis.lang.hierarchy import detect_plants

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _paths(m, b):
    return sorted(m.flat.instances[i].path for i in b.members)


def _find(root, name):
    return next(b for b in root.walk() if b.name == name)


@pytest.fixture(scope="module")
def works():
    m = compile_model([os.path.join(ROOT, "scenarios", "aquila_works", "field.plx")], sim="corp96")
    return m, find_structure(m)


def test_the_blocks_the_plan_uses_are_the_ones_at_their_own_step(works):
    m, root = works
    used = used_blocks(root)
    assert sorted(b.name for b in used) == ["DE.converter", "DE.hsm_works"]
    assert all(b.step == pytest.approx(0.25) and b.parent_step == pytest.approx(1.0) for b in used)
    # the same members as the planner's split (compile_hierarchy uses detect_plants)
    found = {f.name: sorted(f.members) for f in detect_plants(m)}
    assert {b.name: sorted(b.members) for b in used} == found
    conv = _find(root, "DE.converter")
    assert any("converter.tap" in x for x in conv.exact) and any("caster_works.run" in x for x in conv.exact)
    assert any("hsm_works.change" in x for x in _find(root, "DE.hsm_works").exact)


def test_networks_sites_and_lines_come_out_of_the_cut_rules(works):
    m, root = works
    # the group splits at the markets several networks share
    assert root.rule == "exchanges" and len(root.children) >= 8
    eu = next(b for b in root.children if "DE+FR+NL" in b.name)
    assert eu.rule == "time scale"                   # the Duisburg blocks at 15 minutes inside the European network
    # a site splits into its lines at its grid connection (shared by them, no stock): Velsen's iron and steel chain,
    # its strip mill and its coating line
    nl = next(b for b in eu.children if b.name.startswith("NL."))
    assert nl.rule == "exchanges"
    kids = [_paths(m, c) for c in nl.children]
    assert ["NL_HSM"] in kids and ["NL_CGL"] in kids
    assert any("NL_GRID" in w for w in nl.why)


def test_the_oil_field_splits_into_its_pads_at_the_batteries():
    m = compile_model([os.path.join(ROOT, "scenarios", "kestrel", "field.plx")])
    root = find_structure(m)
    assert root.rule == "stocks"
    assert {w.split(":")[0] for w in root.why} == {"CTB_N", "CTB_C", "CTB_S"}
    pads = [b for b in root.children if b.name.split(".")[0].startswith("P") and b.name.endswith("esp_well")]
    assert len(pads) == 12 and not used_blocks(root)


def test_a_twin_with_nothing_to_split_finer_uses_no_block():
    m = compile_model([os.path.join(ROOT, "scenarios", "corvus", "field.plx")])
    root = find_structure(m)
    assert not used_blocks(root)
