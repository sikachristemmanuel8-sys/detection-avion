"""Lecture des annotations au format Pascal VOC (fichiers .xml).

Chaque image `images/<id>.jpg` a une annotation `annotations/<id>.xml` qui liste
les avions présents : nom de la classe (A1..A20) + boîte englobante
(xmin, ymin, xmax, ymax) en pixels.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "..", "..", "student-dataset 2")


@dataclass
class Box:
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    truncated: bool = False

    @property
    def w(self) -> float:
        return self.xmax - self.xmin

    @property
    def h(self) -> float:
        return self.ymax - self.ymin


@dataclass
class ImageAnnotation:
    image_id: str
    image_path: str
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)


def _text(node, tag, default=None):
    child = node.find(tag)
    return child.text if child is not None and child.text is not None else default


def parse_xml(xml_path: str, images_dir: str) -> ImageAnnotation:
    root = ET.parse(xml_path).getroot()
    image_id = os.path.splitext(os.path.basename(xml_path))[0]
    size = root.find("size")
    ann = ImageAnnotation(
        image_id=image_id,
        image_path=os.path.join(images_dir, image_id + ".jpg"),
        width=int(_text(size, "width")),
        height=int(_text(size, "height")),
    )
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        ann.boxes.append(Box(
            label=_text(obj, "name").strip(),
            xmin=float(_text(b, "xmin")),
            ymin=float(_text(b, "ymin")),
            xmax=float(_text(b, "xmax")),
            ymax=float(_text(b, "ymax")),
            truncated=_text(obj, "truncated", "0") != "0",
        ))
    return ann


def load_dataset(dataset_dir: str = DEFAULT_DATASET):
    """Charge toutes les annotations dont l'image existe.

    Retourne (annotations, orphelins) : les XML sans image correspondante
    (ex. `3 - Copie.xml`) sont écartés et listés à part.
    """
    ann_dir = os.path.join(dataset_dir, "annotations")
    img_dir = os.path.join(dataset_dir, "images")
    annotations, orphans = [], []
    for name in sorted(os.listdir(ann_dir)):
        if not name.endswith(".xml"):
            continue
        ann = parse_xml(os.path.join(ann_dir, name), img_dir)
        if os.path.exists(ann.image_path):
            annotations.append(ann)
        else:
            orphans.append(name)
    return annotations, orphans


def class_sort_key(label: str):
    """Trie A1, A2, ..., A10 dans l'ordre numérique (et pas A1, A10, A11...)."""
    digits = "".join(c for c in label if c.isdigit())
    return (int(digits) if digits else 10**9, label)
