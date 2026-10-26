# Проверка реализации — 25 октября 2026

`uv run --no-sync pytest -q --junitxml=runs/validation_20261025.xml`: **127 passed**, 47.62 s. Машиночитаемый результат: [JUnit XML](runs/validation_20261025.xml).

`ruff check`, `ruff format --check` и `uv lock --check` успешны. В lock зафиксирована 121 зависимость с учётом optional extras.

Проверены classification, BYOL, MAE и segmentation fit/restore; обязательные мультимодальные input contracts; optional labels/permissions; trainable custom objectives; distributed masked gradients; deterministic/weighted/PK samplers; MixUp/CutMix; TTA; freeze/unfreeze; scheduler factories и warmup; top-K min/max, ties, NaN, periodic snapshots и сохранение старых файлов; два train/val/test manifests; test cadence; kNN/probe; ансамбли, prepared CV/OOF; offline reports и исходные размеры segmentation.

Epoch resume с двумя DataLoader workers совпадает по augmentation hashes, model/optimizer state и LR. Reproduce из исходного и resumed run совпадает по весам и optimizer state, включая dropout и trainable ArcFace objective. Смена batch recipe, content данных или class order обнаруживается. Включение test observers и kNN/probe сохраняет training trajectory/RNG.

Фактическая проверка GPU: две NVIDIA RTX 3090, PyTorch 2.8.0, Lightning 2.5.6, CUDA 12.8. Команда: `uv run --no-sync python examples/verify_gpu.py`.

- Непрерывный DDP run: [result.json](runs/run_20261025T194901_3fed8ef9d9/result.json).
- Stop/strict resume: [result.json](runs/resume_20261025T194924_e17796163a/result.json).
- Model state, optimizer state и scheduler совпали точно.
- AMP с accumulation: [result.json](runs/run_20261025T194935_270ff347aa/result.json); два успешных updates.
- Отдельный CUDA-тест искусственного overflow подтвердил: scaler пропускает update, scheduler и BYOL EMA не продвигаются, scaler state сохраняется.
- На данных, не кратных world size, GPU DDP и standalone evaluation совпали по counts/accuracy/F1; технический padding не добавляет наблюдения.

Timm, torchvision, Hugging Face ViT/SegFormer, HF Datasets, COCO polygons, Albumentations и Kornia проверены локально. Восстановление HF checkpoints проверено с запрещёнными повторными вызовами `from_pretrained`. Реальное сетевое скачивание пользовательских pretrained weights в тесты не включено.

Предупреждения pytest относятся к CPU-only тестам на GPU-сервере, числу workers, намеренно замороженному BYOL teacher и deprecation предупреждениям multiprocessing/pycocotools. Ошибок тестов нет.

Strict mid-epoch resume, multi-node/FSDP, hierarchical task и model soup остаются расширениями согласно ТЗ. Файлы и результаты проверок сохранены; автоматического удаления checkpoints/caches нет. Особенности обновления имени `last.ckpt` при запрете удаления описаны в README.
