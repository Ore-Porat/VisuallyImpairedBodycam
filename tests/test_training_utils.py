"""Unit tests for the pure parts of training/ (no network, torch or ultralytics needed).

What is a test file?
    pytest runs every function whose name starts with ``test_``. Each one checks one
    behavior and fails loudly if it is wrong.

What is being tested here?
    The scripts in training/ prepare data and tune the object-detection model. Most of
    their work needs a GPU, a dataset download and the ultralytics library, but the
    small "pure" helper functions (text parsing, math, file formatting) can be tested
    here on any machine:
      * get_grocereye_data.py  (imported as ``gd``): cleaning class names, converting
        Pascal VOC XML box annotations into YOLO text labels, splitting a dataset into
        train / val / test sets, and writing config/classes.yaml.
      * train.py               (imported as ``tr``): merging several datasets whose
        class lists differ, by renumbering the class ids in the label files.
      * pick_threshold.py      (imported as ``pt``): the precision / recall math used
        to choose the detector's confidence threshold, and the code that writes the
        chosen number back into config/server.yaml without disturbing the rest of it.

Words you will see:
    VOC / YOLO    Two label-file formats. VOC is XML with pixel corners (xmin, ymin,
                  xmax, ymax). YOLO is one text line per box: "class cx cy w h", all as
                  fractions 0..1 of the image size.
    ground truth  (GT) The correct, human-labelled boxes.
    TP / FP / FN  True positive (correct detection), false positive (detected something
                  that is not there), false negative (missed a real object).
    IoU           Intersection over union: how much two boxes overlap, 0..1.
    precision     Of everything detected, the fraction that was right.
    recall        Of everything really there, the fraction that was detected.
    parametrize   Runs the same test once per listed input row.
    pytest.approx "Equal, allowing for tiny floating-point rounding differences."
    tmp_path      A temporary folder from pytest, deleted after the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# training/ is not a Python package, so add its folder to the import search path and
# import the scripts as plain modules. "noqa: E402" tells the linter the late import is on purpose.
TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

import get_grocereye_data as gd  # noqa: E402
import pick_threshold as pt  # noqa: E402
import train as tr  # noqa: E402

# A small hand-written VOC annotation file for a 224 x 224 image with three boxes:
#   1. a normal tomato sauce box,
#   2. a "Beans" box that sticks out past the image edges (must be clipped),
#   3. a "beans" box with zero width (must be dropped).
VOC = """<annotation>
  <folder>tomato_sauce</folder>
  <filename>TOMATO_SAUCE0000.png</filename>
  <size><width>224</width><height>224</height><depth>3</depth></size>
  <object><name>TOMATO_SAUCE</name><bndbox>
    <xmin>131</xmin><ymin>5</ymin><xmax>215</xmax><ymax>96</ymax></bndbox></object>
  <object><name>Beans</name><bndbox>
    <xmin>-10</xmin><ymin>100</ymin><xmax>50</xmax><ymax>300</ymax></bndbox></object>
  <object><name>beans</name><bndbox>
    <xmin>10</xmin><ymin>10</ymin><xmax>10</xmax><ymax>40</ymax></bndbox></object>
</annotation>
"""


# ---- class names ------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("TOMATO_SAUCE", "tomato_sauce"), ("tomato sauce", "tomato_sauce"), ("  Beans ", "beans"),
    ("Tomato-Sauce", "tomato_sauce"), ("cereal", "cereal"), ("Honey Nut  Cheerios!", "honey_nut_cheerios"),
])
def test_normalize_class_name(raw, expected):
    """Messy dataset folder names become clean lowercase ids with underscores."""
    assert gd.normalize_class_name(raw) == expected


def test_normalize_class_name_empty():
    """A name with no letters or digits left after cleaning is an error."""
    with pytest.raises(ValueError):
        gd.normalize_class_name(" -- ")


def test_spoken_default():
    """The default spoken name is the id with underscores turned into spaces."""
    assert gd.spoken_default("tomato_sauce") == "tomato sauce"


# ---- VOC -> YOLO --------------------------------------------------------------------------

def test_voc_xml_to_yolo(tmp_path):
    """VOC pixel boxes become YOLO fractions; boxes are clipped to the image or dropped."""
    xml = tmp_path / "TOMATO_SAUCE0000.xml"
    xml.write_text(VOC, encoding="utf-8")
    ann = gd.parse_voc_xml(xml.read_text(encoding="utf-8"))
    assert ann.filename == "TOMATO_SAUCE0000.png"
    assert (ann.width, ann.height) == (224, 224)
    assert [b.name for b in ann.boxes] == ["tomato_sauce", "beans", "beans"]

    # The dict maps class id -> class number used in the YOLO label file.
    lines = gd.voc_to_yolo_lines(ann, {"beans": 0, "tomato_sauce": 1})
    assert len(lines) == 2  # zero-width box dropped
    c, cx, cy, w, h = lines[0].split()
    assert c == "1"
    # Center and size as fractions of the 224-pixel image.
    assert float(cx) == pytest.approx((131 + 215) / 2 / 224, abs=1e-6)
    assert float(cy) == pytest.approx((5 + 96) / 2 / 224, abs=1e-6)
    assert float(w) == pytest.approx(84 / 224, abs=1e-6)
    assert float(h) == pytest.approx(91 / 224, abs=1e-6)
    # second box is clipped to the image: x 0..50, y 100..224
    c, cx, cy, w, h = map(float, lines[1].split())
    assert c == 0
    assert (cx, w) == (pytest.approx(25 / 224, abs=1e-6), pytest.approx(50 / 224, abs=1e-6))
    assert (cy, h) == (pytest.approx(162 / 224, abs=1e-6), pytest.approx(124 / 224, abs=1e-6))
    assert all(0 <= v <= 1 for line in lines for v in map(float, line.split()[1:]))


def test_voc_size_override_when_missing():
    """If the XML has no image size, the caller must supply one or get an error."""
    ann = gd.parse_voc_xml("<annotation><filename>a.png</filename><object><name>milk</name>"
                           "<bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>100</ymax>"
                           "</bndbox></object></annotation>")
    assert ann.width == 0
    with pytest.raises(ValueError):
        gd.voc_to_yolo_lines(ann, {"milk": 0})
    assert gd.voc_to_yolo_lines(ann, {"milk": 0}, 100, 100) == ["0 0.250000 0.500000 0.500000 1.000000"]


def test_find_samples_pairs_images_and_detects_split(tmp_path):
    """Each XML is paired with its image and tagged with the split folder it came from."""
    # Build a tiny fake dataset: dataset/{train,val}/{annotations,images}/beans/...
    for split in ("train", "val"):
        (tmp_path / "dataset" / split / "annotations" / "beans").mkdir(parents=True)
        (tmp_path / "dataset" / split / "images" / "beans").mkdir(parents=True)
        (tmp_path / "dataset" / split / "annotations" / "beans" / f"B_{split}.xml").write_text(
            VOC.replace("TOMATO_SAUCE0000.png", f"B_{split}.png"), encoding="utf-8")
        (tmp_path / "dataset" / split / "images" / "beans" / f"B_{split}.png").write_bytes(b"x")
    samples = gd.find_samples(tmp_path)
    assert sorted((s.key, s.source_split) for s in samples) == [("B_train", "train"), ("B_val", "val")]
    # The "stratum" (group used to balance the split) is the class folder name.
    assert all(s.stratum == "beans" for s in samples)


# ---- split ----------------------------------------------------------------------------------

def _items():
    """157 fake samples: 50 each of beans / milk / tea plus 7 corn, as (key, class) pairs."""
    return [(f"{c}{i:04d}", c) for c in ("beans", "milk", "tea") for i in range(50)] + \
           [(f"corn{i:04d}", "corn") for i in range(7)]


def test_split_deterministic_and_order_independent():
    """The same seed always gives the same split, whatever order the items came in."""
    a = gd.stratified_split(_items(), 0.1, 0.1, seed=0)
    b = gd.stratified_split(list(reversed(_items())), 0.1, 0.1, seed=0)
    assert a == b
    c = gd.stratified_split(_items(), 0.1, 0.1, seed=1)
    assert c != a


def test_split_fractions_and_disjoint():
    """10% val and 10% test PER CLASS, no sample in two sets, nothing lost."""
    s = gd.stratified_split(_items(), 0.1, 0.1, seed=0)
    allk = s["train"] + s["val"] + s["test"]
    assert len(allk) == len(set(allk)) == len(_items())
    for cls in ("beans", "milk", "tea"):
        assert sum(k.startswith(cls) for k in s["test"]) == 5
        assert sum(k.startswith(cls) for k in s["val"]) == 5
    assert sum(k.startswith("corn") for k in s["test"]) == 1  # round(0.7)


def test_split_rejects_bad_fractions():
    """val + test fractions that leave nothing for training are refused."""
    with pytest.raises(ValueError):
        gd.stratified_split(_items(), 0.5, 0.5, seed=0)


# ---- classes.yaml -------------------------------------------------------------------------

def test_classes_yaml_keeps_hand_edits(tmp_path):
    """Regenerating classes.yaml keeps the header comment and hand-edited spoken names."""
    f = tmp_path / "classes.yaml"
    f.write_text("# my header\nclasses:\n  beans: baked beans\n  old_class: gone\n", encoding="utf-8")
    existing = gd.load_spoken_names(f)
    text = gd.render_classes_yaml(["beans", "tomato_sauce"], existing,
                                  gd.leading_comment_block(f.read_text(encoding="utf-8")))
    # "baked beans" survives, "old_class" (no longer in the dataset) is dropped.
    assert text == "# my header\nclasses:\n  beans: baked beans\n  tomato_sauce: tomato sauce\n"
    assert yaml.safe_load(text)["classes"] == {"beans": "baked beans", "tomato_sauce": "tomato sauce"}


def test_classes_yaml_default_format():
    """With no existing file, the generated YAML has the standard header and defaults."""
    text = gd.render_classes_yaml(["beans", "tomato_sauce"])
    assert text == ("# GrocerEye / Freiburg Groceries classes: class_id -> spoken name (FR-62).\n"
                    "classes:\n  beans: beans\n  tomato_sauce: tomato sauce\n")


def test_classes_yaml_quotes_awkward_names():
    """Names that YAML would misread (yes/no, colons, apostrophes) are quoted correctly."""
    text = gd.render_classes_yaml(["no", "jam"], {"jam": "Mom's: jam"})
    assert yaml.safe_load(text)["classes"] == {"no": "no", "jam": "Mom's: jam"}


# ---- combined data yaml / remap ------------------------------------------------------------

def test_class_remap_and_label_rewrite():
    """A dataset's class numbers are translated into the combined class list's numbers."""
    # Dataset classes ["Tomato Sauce", "beans"] -> combined ["beans", "cake", "tomato_sauce"]:
    # dataset 0 (tomato sauce) becomes 2, dataset 1 (beans) becomes 0.
    remap = tr.class_remap(["Tomato Sauce", "beans"], ["beans", "cake", "tomato_sauce"])
    assert remap == {0: 2, 1: 0}
    assert tr.remap_label_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n", remap) == \
        "2 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n"
    # A class that is not in the combined list is an error, not silently dropped.
    with pytest.raises(ValueError):
        tr.class_remap(["ramen"], ["beans"])


def test_combine_data_yamls(tmp_path):
    """Two datasets with different class lists merge into one data.yaml with remapped labels."""
    def make(root: Path, names: list[str], label: str):
        """Create a minimal YOLO dataset folder with one image + label per split."""
        for s in ("train", "val"):
            (root / "images" / s).mkdir(parents=True)
            (root / "labels" / s).mkdir(parents=True)
            (root / "images" / s / f"{root.name}_{s}.jpg").write_bytes(b"x")
            (root / "labels" / s / f"{root.name}_{s}.txt").write_text(label, encoding="utf-8")
        (root / "data.yaml").write_text(yaml.safe_dump(
            {"train": "images/train", "val": "images/val", "names": names}), encoding="utf-8")
        return root / "data.yaml"

    a = make(tmp_path / "freiburg", ["beans", "tea"], "1 0.5 0.5 0.2 0.2\n")
    b = make(tmp_path / "ours", ["tea"], "0 0.5 0.5 0.2 0.2\n")  # Roboflow-style subset
    out = tr.combine_data_yamls([a, b], tmp_path / "combined.yaml", remap_root=tmp_path / "remapped")
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["names"] == ["beans", "tea"] and doc["nc"] == 2
    assert len(doc["train"]) == 2 and len(doc["val"]) == 2
    # In "ours", tea was class 0; in the combined list tea is class 1.
    remapped_label = tmp_path / "remapped" / "ours" / "labels" / "train" / "ours_train.txt"
    assert remapped_label.read_text(encoding="utf-8") == "1 0.5 0.5 0.2 0.2\n"


def test_label_path_for():
    """images/train/a.jpg has its label at labels/train/a.txt (the YOLO convention)."""
    p = Path("data") / "x" / "images" / "train" / "a.jpg"
    assert tr.label_path_for(p) == Path("data") / "x" / "labels" / "train" / "a.txt"


# ---- threshold sweep math -------------------------------------------------------------------

def test_iou():
    """Identical boxes overlap fully (1.0), touching corners not at all (0.0)."""
    assert pt.iou((0, 0, 1, 1), (0, 0, 1, 1)) == pytest.approx(1.0)
    assert pt.iou((0, 0, 1, 1), (1, 1, 2, 2)) == 0.0
    assert pt.iou((0, 0, 2, 1), (1, 0, 3, 1)) == pytest.approx(1 / 3)


def test_match_image_basic():
    """Predictions are matched to ground truth one-to-one; extras and wrong classes are FPs."""
    # Ground truth boxes: (class, x1, y1, x2, y2).
    gts = [(0, 0.0, 0.0, 0.5, 0.5), (1, 0.5, 0.5, 1.0, 1.0)]
    # Predictions: (class, confidence, x1, y1, x2, y2).
    preds = [
        (0, 0.9, 0.0, 0.0, 0.5, 0.5),    # TP class 0
        (0, 0.8, 0.0, 0.0, 0.5, 0.5),    # duplicate -> FP class 0
        (2, 0.7, 0.5, 0.5, 1.0, 1.0),    # wrong class on GT 1 -> FP class 2, GT 1 FN
        (1, 0.3, 0.5, 0.5, 1.0, 1.0),    # TP class 1 but only below 0.5
    ]
    # At threshold 0.5 the 0.3-confidence prediction is ignored.
    hi = pt.match_image(preds, gts, conf_thr=0.5)
    assert (hi[0].tp, hi[0].fp, hi[0].fn) == (1, 1, 0)
    assert (hi[1].tp, hi[1].fp, hi[1].fn) == (0, 0, 1)
    assert (hi[2].tp, hi[2].fp, hi[2].fn) == (0, 1, 0)
    # At threshold 0.2 it counts, so class 1 is found.
    lo = pt.match_image(preds, gts, conf_thr=0.2)
    assert (lo[1].tp, lo[1].fn) == (1, 0)


def test_match_requires_iou():
    """A prediction that barely overlaps the true box is a miss AND a false alarm."""
    gts = [(0, 0.0, 0.0, 0.2, 0.2)]
    preds = [(0, 0.9, 0.1, 0.1, 0.3, 0.3)]  # IoU = 0.01/0.07 ~ 0.14
    r = pt.match_image(preds, gts, 0.5, iou_thr=0.5)
    assert (r[0].tp, r[0].fp, r[0].fn) == (0, 1, 1)


def test_thresholds_range():
    """0.20, 0.25, ... 0.95 as clean decimals (16 values)."""
    t = pt.thresholds_range(0.2, 0.95, 0.05)
    assert t[0] == 0.2 and t[-1] == 0.95 and len(t) == 16
    assert 0.35 in t  # no float drift like 0.35000000000000003


def _sweep_fixture():
    """A small made-up evaluation set: list of (predictions, ground_truths) per image."""
    box = (0.1, 0.1, 0.6, 0.6)
    images = []
    for _ in range(9):  # 9 images of class 0, detected at 0.8
        images.append(([(0, 0.8, *box)], [(0, *box)]))
    # a background image where class 0 is hallucinated at 0.45, class 1 at 0.9
    images.append(([(0, 0.45, *box), (1, 0.9, *box)], []))
    # a class-1 image detected at 0.6
    images.append(([(1, 0.6, *box)], [(1, *box)]))
    return images


def test_sweep_and_recommend():
    """Sweep every threshold, then recommend the lowest one where precision is good enough."""
    images = _sweep_fixture()
    res = pt.sweep(images, pt.thresholds_range(0.2, 0.95, 0.05))
    # at 0.40: class 0 P = 9/10 = 0.9 (passes), class 1 P = 1/2 (fails)
    assert res[0.4][0].precision == pytest.approx(0.9)
    assert res[0.4][1].precision == pytest.approx(0.5)
    # restricted to class 0: lowest threshold with P >= 0.9 is the first one
    assert pt.recommend_threshold(res, [0], 0.90) == 0.2
    assert pt.recommend_threshold(res, [0], 0.95) == 0.5
    # class 1 has a 0.9-confidence false positive, so it never reaches P >= 0.9 while it has a TP;
    # above 0.9 it has no predictions (undefined precision counts as passing)
    assert pt.recommend_threshold(res, [0, 1], 0.90) == 0.95
    assert pt.recommend_threshold(res, [0, 1], 0.90, min_recall=0.5) is None
    assert pt.failing_classes(res[0.5], [0, 1], 0.9) == [1]


def test_confusion_matrix():
    """Rows = true class (last row = background), columns = predicted (last = missed)."""
    box_a, box_b = (0.0, 0.0, 0.4, 0.4), (0.5, 0.5, 0.9, 0.9)
    images = [
        ([(0, 0.9, *box_a), (1, 0.9, *box_b)], [(0, *box_a), (0, *box_b)]),  # 0->0, 0->1
        ([(1, 0.9, *box_a)], []),                                            # bg -> 1
        ([], [(1, *box_b)]),                                                 # 1 -> missed
        ([(0, 0.1, *box_a)], [(0, *box_a)]),                                 # below conf: missed
    ]
    m = pt.confusion_matrix(images, nc=2, conf_thr=0.5)
    assert m == [[1, 1, 1],
                 [0, 0, 1],
                 [0, 1, 0]]


def test_read_yolo_labels(tmp_path):
    """YOLO 'class cx cy w h' lines are read back as corner boxes; a missing file is empty."""
    f = tmp_path / "a.txt"
    f.write_text("3 0.5 0.5 0.2 0.4\n\n", encoding="utf-8")
    [(c, x1, y1, x2, y2)] = pt.read_yolo_labels(f)
    assert c == 3 and (x1, y1, x2, y2) == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert pt.read_yolo_labels(tmp_path / "missing.txt") == []


# ---- server.yaml in-place edit ----------------------------------------------------------------

# A pretend server.yaml. The decoy conf_find keys in other sections must NOT be touched;
# only the one under "guidance:" is the real setting.
SERVER = """# Laptop server configuration
model:
  conf_find: 0.99        # decoy in another section: must not change
  min_conf: 0.25

guidance:
  # Confidence thresholds. conf_find is rewritten by training/pick_threshold.py (TR-14).
  conf_find: 0.5    # keep me
  conf_identify: 0.6

  stable_k: 3

tts:
  conf_find: 0.11
"""


def test_set_conf_find_in_place(tmp_path):
    """Only the guidance conf_find value changes; comments and other sections stay as they are."""
    f = tmp_path / "server.yaml"
    f.write_text(SERVER, encoding="utf-8")
    new = pt.set_conf_find(f.read_text(encoding="utf-8"), 0.55)
    expected = SERVER.replace("  conf_find: 0.5    # keep me", "  conf_find: 0.55    # keep me")
    assert new == expected
    data = yaml.safe_load(new)
    assert data["guidance"]["conf_find"] == 0.55
    assert data["model"]["conf_find"] == 0.99 and data["tts"]["conf_find"] == 0.11


def test_set_conf_find_formats_and_crlf():
    """Windows line endings (CRLF) are preserved, and whole numbers are written as floats."""
    text = SERVER.replace("\n", "\r\n")
    new = pt.set_conf_find(text, 0.7)
    assert "  conf_find: 0.7    # keep me\r\n" in new
    assert new.count("\r\n") == text.count("\r\n")
    assert pt.set_conf_find("guidance:\n  conf_find: 0.5\n", 1) == "guidance:\n  conf_find: 1.0\n"


def test_set_conf_find_inserts_when_missing():
    """If guidance has no conf_find yet, one is inserted with matching indentation."""
    new = pt.set_conf_find("guidance:\n    stable_k: 3\nother: 1\n", 0.65)
    assert new == "guidance:\n    conf_find: 0.65\n    stable_k: 3\nother: 1\n"
    # No guidance section at all: refuse rather than guess.
    with pytest.raises(ValueError):
        pt.set_conf_find("model:\n  x: 1\n", 0.5)


def test_set_conf_find_on_real_server_yaml():
    """The edit works on the real config/server.yaml and changes exactly one line."""
    real = Path(__file__).resolve().parent.parent / "config" / "server.yaml"
    if not real.exists():
        pytest.skip("config/server.yaml missing")
    text = real.read_text(encoding="utf-8")
    new = pt.set_conf_find(text, 0.65)
    diff = [(a, b) for a, b in zip(text.splitlines(), new.splitlines()) if a != b]
    assert len(diff) == 1 and "conf_find" in diff[0][0]
    assert yaml.safe_load(new)["guidance"]["conf_find"] == 0.65
