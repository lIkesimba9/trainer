"""Portable reports rebuilt from predictions without a model or accelerator."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from trainer_project.evaluation.accumulator import (
    EvaluationAccumulator,
    json_value,
    project_path,
    read_predictions,
)


def _dump(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(json_value(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)


def _confusion_table(details: dict) -> str:
    matrix = details.get("confusion_matrix", [])
    names = details.get("class_names", [])
    if not matrix:
        return ""
    table = '<table class="confusion"><thead><tr><th>True / predicted</th>'
    table += "".join(f"<th>{html.escape(str(name))}</th>" for name in names) + "</tr></thead><tbody>"
    maximum = max(max(row) for row in matrix) or 1
    for name, values in zip(names, matrix):
        table += f"<tr><th>{html.escape(str(name))}</th>"
        for value in values:
            opacity = 0.08 + 0.55 * value / maximum
            table += f'<td style="background:rgba(35,109,190,{opacity:.3f})">{value}</td>'
        table += "</tr>"
    return table + "</tbody></table>"


def _curves(rows: list[dict], field: str, names: list[str]) -> str:
    valid = [
        row
        for row in rows
        if field in row.get("probabilities", {})
        and bool(np.asarray(row.get("target_valid", {}).get(field, False)).all())
        and np.asarray(row.get("targets", {}).get(field, [])).ndim == 0
    ]
    if not valid:
        return ""
    try:
        from sklearn.metrics import precision_recall_curve, roc_curve
    except ImportError:
        return ""
    targets = np.asarray([row["targets"][field] for row in valid])
    probs = np.asarray([row["probabilities"][field] for row in valid])
    charts = []
    palette = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706", "#0891b2"]
    for title, function in (("ROC", roc_curve), ("Precision–recall", precision_recall_curve)):
        paths, legend = [], []
        for index, name in enumerate(names):
            binary = targets == index
            if not binary.any() or not (~binary).any():
                continue
            first, second, _ = function(binary, probs[:, index])
            x, y = (second, first) if title == "Precision–recall" else (first, second)
            points = " ".join(f"{30 + 300 * a:.2f},{320 - 300 * b:.2f}" for a, b in zip(x, y))
            color = palette[index % len(palette)]
            paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
            legend.append(f'<span style="color:{color}">{html.escape(name)}</span>')
        if paths:
            charts.append(
                f'<figure><figcaption>{title}</figcaption><svg viewBox="0 0 350 350" width="350" height="350" role="img" aria-label="{title}">'
                '<path d="M30 20V320H330" fill="none" stroke="#777"/>'
                + "".join(paths)
                + "</svg><p>"
                + ", ".join(legend)
                + "</p></figure>"
            )
    return '<div class="curves">' + "".join(charts) + "</div>" if charts else ""


def _export_masks(rows: list[dict], output: Path) -> list[dict]:
    from PIL import Image

    exported = []
    for row in rows:
        if "predicted_mask" not in row:
            continue
        array = np.asarray(row["predicted_mask"])
        if array.ndim != 2 or array.max(initial=0) > 254:
            raise ValueError("Indexed uint8 PNG export supports exclusive class IDs 0..254")
        digest = hashlib.sha256(
            f"{row['role']}/{row['manifest_name']}/{row['sample_id']}".encode()
        ).hexdigest()[:16]
        directory = output / "masks"
        directory.mkdir(exist_ok=True)
        mask_path = directory / f"{digest}.png"
        with mask_path.open("xb") as stream:
            Image.fromarray(array.astype(np.uint8)).save(stream, format="PNG")
        item = {"sample_id": row["sample_id"], "mask": str(mask_path.relative_to(output))}
        image_path = row.get("metadata", {}).get("image_path", row.get("metadata", {}).get("image"))
        if isinstance(image_path, str):
            try:
                source = project_path(image_path)
            except ValueError:
                source = None
            if source is not None and source.is_file():
                with Image.open(source) as image:
                    image = image.convert("RGB").resize((array.shape[1], array.shape[0]))
                    colors = np.stack(
                        ((array * 67) % 256, (array * 131 + 50) % 256, (array * 197 + 90) % 256), axis=-1
                    ).astype(np.uint8)
                    overlay = Image.blend(image, Image.fromarray(colors), 0.4)
                    overlay_path = directory / f"{digest}_overlay.png"
                    with overlay_path.open("xb") as stream:
                        overlay.save(stream, format="PNG")
                    item["overlay"] = str(overlay_path.relative_to(output))
        exported.append(item)
    return exported


def build_report(
    records: Iterable[dict] | str | Path,
    output_dir: str | Path,
    *,
    task: str | None = None,
    class_maps: dict | None = None,
    target_key: str | None = None,
    provenance: dict | None = None,
    pooled_policy: str | None = None,
    identity_key: str | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    rows = read_predictions(records) if isinstance(records, (str, Path)) else list(records)
    output = project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot rebuild a report from empty predictions")
    task = task or rows[0].get("task", "classification")
    class_maps = class_maps or rows[0].get("class_maps", {})
    saved_protocol = rows[0].get("evaluation_protocol", {})
    target_key = target_key or saved_protocol.get("target_key", "label")
    pooled_policy = pooled_policy or saved_protocol.get("pooled_policy", "observations")
    identity_key = identity_key or saved_protocol.get("identity_key", "sample_id")
    for name in ("ignore_index", "zero_division", "empty_mask_policy", "manifest_macro", "slice_fields"):
        if name in saved_protocol:
            kwargs.setdefault(name, saved_protocol[name])
    evaluator = EvaluationAccumulator(
        task,
        class_maps,
        target_key,
        pooled_policy=pooled_policy,
        identity_key=identity_key,
        provenance=provenance,
        **kwargs,
    )
    evaluator.add_records(rows)
    metrics = evaluator.compute()
    payload = {
        "task": task,
        "metrics": metrics,
        "details": evaluator.details,
        "provenance": provenance or {},
        "prediction_provenance": [row.get("provenance", {}) for row in rows],
        "protocol_hash": evaluator.protocol_hash(),
        "records": len(rows),
    }
    _dump(output / "report.json", payload)
    with (output / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "role",
                "manifest_name",
                "sample_id",
                "target_field",
                "target",
                "target_valid",
                "prediction",
                "probabilities",
                "metadata",
            ],
        )
        writer.writeheader()
        for row in rows:
            for field, probabilities in row.get("probabilities", {}).items():
                writer.writerow(
                    {
                        "role": row["role"],
                        "manifest_name": row["manifest_name"],
                        "sample_id": row["sample_id"],
                        "target_field": field,
                        "target": json.dumps(row.get("targets", {}).get(field)),
                        "target_valid": json.dumps(row.get("target_valid", {}).get(field, False)),
                        "prediction": int(np.argmax(probabilities)),
                        "probabilities": json.dumps(probabilities),
                        "metadata": json.dumps(row.get("metadata", {}), ensure_ascii=False),
                    }
                )
    sections = []
    for name, detail in evaluator.details.items():
        section = f"<section><h2>{html.escape(name)}</h2><p>Observations: {detail.get('observations', 0)}</p>"
        for field, values in detail.items():
            if not isinstance(values, dict):
                continue
            section += f"<h3>{html.escape(field)}</h3>"
            scalar = {
                key: value
                for key, value in values.items()
                if isinstance(value, (int, float)) or value is None
            }
            section += (
                "<dl>"
                + "".join(
                    f"<dt>{html.escape(key)}</dt><dd>{value if value is not None else 'N/A'}</dd>"
                    for key, value in scalar.items()
                )
                + "</dl>"
            )
            section += _confusion_table(values)
            role, manifest = name.split("/", 1)
            selected = [
                row
                for row in rows
                if row["role"] == role and (manifest == "pooled" or row["manifest_name"] == manifest)
            ]
            section += _curves(selected, field, values.get("class_names", []))
        sections.append(section + "</section>")
    errors = []
    for row in rows:
        for field, probabilities in row.get("probabilities", {}).items():
            target = row.get("targets", {}).get(field)
            valid = bool(np.asarray(row.get("target_valid", {}).get(field, False)).all())
            if valid and np.asarray(target).ndim == 0 and target != int(np.argmax(probabilities)):
                errors.append(
                    {
                        "sample_id": row["sample_id"],
                        "manifest": row["manifest_name"],
                        "target_field": field,
                        "target": target,
                        "prediction": int(np.argmax(probabilities)),
                    }
                )
    if errors:
        sections.append(
            "<section><h2>Examples of errors</h2><pre>"
            + html.escape(json.dumps(errors[:30], ensure_ascii=False, indent=2))
            + "</pre></section>"
        )
    masks = _export_masks(rows, output)
    if masks:
        _dump(output / "masks.json", masks)
        figures = "".join(
            f'<figure><figcaption>{html.escape(item["sample_id"])}</figcaption><img width="240" src="{html.escape(item.get("overlay", item["mask"]))}" alt="Segmentation prediction"></figure>'
            for item in masks[:30]
        )
        sections.append("<section><h2>Segmentation examples</h2>" + figures + "</section>")
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Trainer evaluation report</title><style>body{font:16px system-ui;margin:2rem auto;padding:0 1rem;max-width:1100px;color:#172334;background:#fafbfd}section{background:white;padding:1.5rem;margin:1rem 0;border:1px solid #d7e0ea;border-radius:10px}table{border-collapse:collapse}td,th{border:1px solid #d7e0ea;padding:.55rem}dl{display:grid;grid-template-columns:220px 1fr;gap:.4rem}dd{margin:0}.curves{display:flex;flex-wrap:wrap}figure{display:inline-block;margin:1rem}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Trainer evaluation report</h1>'
    document += f"<p>Task: {html.escape(task)} · Predictions: {len(rows)} · Protocol: {html.escape(evaluator.protocol_hash()[:16])}</p>"
    document += "".join(sections)
    document += (
        "<section><h2>Experiment provenance</h2><pre>"
        + html.escape(json.dumps(provenance or {}, ensure_ascii=False, indent=2))
        + "</pre></section></html>"
    )
    with (output / "report.html").open("x", encoding="utf-8") as stream:
        stream.write(document)
    return {
        "html": str(output / "report.html"),
        "json": str(output / "report.json"),
        "csv": str(output / "predictions.csv"),
    }
