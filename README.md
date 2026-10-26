# Trainer: Lightning + Hydra + uv

Трейнер для классификации изображений, BYOL, MAE, смешанных objectives и semantic segmentation. Все данные, кэш, временные файлы, checkpoints и отчёты находятся внутри этого проекта. Программа запрещает пути за его пределами. Исходное ТЗ и `AGENT.md` сохранены.

## Быстрый запуск

Рабочая папка: `/home/simba9/workflow/isp/trainer`. Python 3.12, зависимости зафиксированы в `uv.lock`.

```bash
uv sync --locked --inexact
uv run --no-sync trainer --help
uv run --no-sync trainer train --config examples/classification.yaml
```

Демонстрационные изображения и manifests уже созданы в `examples/demo_data`. Генератор `examples/generate_demo.py` создаёт новый набор и отказывается перезаписывать существующую папку. В demo используются два train, два val и два test manifests.

Дополнительные интеграции:

```bash
uv sync --locked --inexact --all-extras
# Или выборочно: --extra vision --extra augmentations --extra data --extra wandb
```

`--inexact` сохраняет уже установленные дополнительные пакеты, `--no-sync` запускает команду без изменения окружения. Кэш `uv` расположен в `.cache/uv`, окружение — в `.venv`. Зависимости с CUDA установлены и проверены на двух RTX 3090.

## Обучение

```bash
uv run --no-sync trainer validate-data --config examples/classification.yaml
uv run --no-sync trainer train --config examples/classification.yaml trainer.max_epochs=5
uv run --no-sync trainer train --config examples/classification.yaml +experiment=byol
uv run --no-sync trainer train --config examples/classification.yaml +experiment=mae
uv run --no-sync trainer train --config examples/classification.yaml +experiment=segmentation
```

GPU, DDP и AMP:

```bash
uv run --no-sync trainer train --config examples/classification.yaml \
  trainer.accelerator=gpu trainer.devices=2 trainer.strategy=ddp trainer.precision=bf16-mixed

uv run --no-sync trainer train --config examples/classification.yaml \
  trainer.accelerator=gpu trainer.precision=16-mixed trainer.accumulate_grad_batches=2
```

`data.batch_size_per_device` — размер batch на устройстве. Samplers сами разделяют train и evaluation между ranks, поэтому `trainer.use_distributed_sampler=false`. Для objectives, не использующих classifier head, и произвольных mixed recipes с неактивными ветвями задайте `trainer.strategy=ddp_find_unused_parameters_true`.

Число microbatches эпохи при accumulation должно делиться на `accumulate_grad_batches`; неполное окно вызывает явную ошибку. Для неполного minibatch можно задать `+data.drop_last=true`. `+trainer.limit_train_batches=2`, `+trainer.limit_val_batches=1` и `+trainer.overfit_batches=2` доступны для диагностики. Проверка NaN/Inf выполняется до backward. Early stopping задаётся блоком `early_stopping` с `monitor`, `mode` и `patience`.

Optimizer groups создаются для backbone, head и objective; norm/bias исключаются из weight decay по умолчанию. LR/decay переопределяются в `parameter_groups.backbone`, `.head`, `.objective`. `freeze_schedule: {model.backbone: 2}` сохраняет веса и BN statistics замороженными в эпохах 0 и 1, затем разрешает обновления с эпохи 2. Параметры остаются в исходных optimizer/DDP groups, а frozen gradients очищаются перед step.

## Данные и schema

Минимальный supervised CSV:

```csv
sample_id,image,label
s001,images/001.png,class_a
s002,images/002.png,
```

Минимальный SSL CSV может содержать только `image`. Если `sample_id` отсутствует, ID выводится детерминированно из объявленных inputs, с проверкой коллизий. Пустое поле означает missing label; `NA` считается обычной строкой, пока явно не включено в `missing_tokens`.

```yaml
data:
  _target_: trainer_project.data.ManifestDataModule
  _recursive_: false
  data_root: examples/my_data
  batch_size_per_device: 16
  num_workers: 2
  schema:
    inputs:
      image: {column: image, type: image}
    targets:
      label:
        column: label_main
        type: categorical
        classes: [class_a, class_b]
        allow_missing: true
      auxiliary:
        column: label_aux
        type: categorical
        classes: [type_x, type_y]
        allow_missing: true
    metadata: [source]
  train:
    - {name: train_a, path: train_a.csv}
    - {name: train_b, path: train_b.csv, supervision_allowed: {label: false, auxiliary: false}}
  val:
    - {name: val_main, path: val.csv}
  test:
    - {name: test_main, path: test.csv}
```

У каждого источника могут быть собственные `root`, `mapping`, `schema` и supervision permissions. Относительные пути к assets разрешаются от `data_root`/корня manifest; schema и class-map файлы — от корня проекта. Schema можно вынести в YAML. Все объявленные labels, validity, permissions, input presence и metadata доходят до custom objective. У каждой categorical метки свой `class_to_idx`; автоматическое определение классов использует только разрешённые train rows.

Train по умолчанию объединяется в один dataset. В `data.train_mode=combined` доступен named CombinedLoader с явным `combined_mode`; общий batch объединяет именованные потоки, сохраняя происхождение samples. Готовые splits задаёт автор: случайных разбиений нет. Отсутствующие/повреждённые изображения, неверные IDs, неизвестные классы и несогласованные masks вызывают ошибку до fit.

`data.identity_policy` управляет пересечениями; доступные поля и политики описаны в [datamodule.py](src/trainer_project/data/datamodule.py). Для использования held-out images в SSL train нужно явно разрешить пересечение и отключить их supervised labels.

## Поддерживаемые контракты

| Recipe / модель | Входы | Objective / выход |
| --- | --- | --- |
| `ImageClassifier` | Image | CE, focal, LDAM, logit adjustment, Seesaw, ArcFace/subcenter, triplet |
| `BYOLModel` | Image + две SSL views | BYOL; mixed ArcFace/triplet через `CompositeObjective` |
| `MAEModel` | Image с фиксированным размером и patch geometry | Reconstruction на masked patches; mixed embedding objectives |
| `MultiModalModel(profile=image_image)` | `image_a`, `image_b` | Fusion, logits, embeddings, per-modality embeddings |
| `MultiModalModel(profile=image_image_text)` | `image_a`, `image_b`, tokenized `text` | То же; cross-modal loss задаётся custom objective |
| `MultiModalModel(profile=image_mask)` | `image`, бинарный `input_mask` | Явный mask-as-channel adapter |
| `TinySegmenter`, `SegFormerAdapter` | Image | `dense_logits`, masked CE/BCE/Dice |

Backbones: встроенный CNN, `TimmBackbone`, `TorchvisionBackbone`, `HuggingFaceBackbone`; pooling/neck/head и компоненты SSL заменяются через `_target_`. Для pretrained timm нужен конкретный weights variant, для HF — полный commit revision, для torchvision — конкретный weights enum. Preprocessing задаётся явно и сохраняется в snapshot. При resume/evaluate исходные pretrained weights повторно не загружаются. HF architecture сохраняется целиком для offline reconstruction.

Встроенные модели требуют присутствия своих входов. Presence-aware fusion и нестандартные способы применения input mask подключаются отдельным классом. BYOL/MAE не создают text alignment автоматически. `compatibility.custom_validated=true` означает явное принятие автором контракта своего adapter/objective; встроенные несовместимые сочетания всё равно проверяются.

Основной segmentation manifest: `sample_id,image,segmentation_mask`. Target — одноканальный lossless PNG с class IDs, например `0=background,1=foreground,255=ignore`. RGB-цвета не подменяют IDs. Spatial validity задаётся `validity_column`; отсутствующая аннотация исключается из loss/metrics. Маски 0/255 требуют явного remap. PNG predictions и метрики возвращаются к оригинальному H×W; original targets/validity сохраняются до resize. Binary/multilabel objectives поддерживают tensor contract; независимые overlapping masks требуют соответствующего dataset adapter.

HF Dataset adapter читает явно выбранный split локального dataset или зафиксированного hub revision. COCO adapter требует `variant: instances|stuff|panoptic`, category mapping, background/ignore и overlap policy. Их параметры описаны в [adapters.py](src/trainer_project/data/adapters.py).

## Аугментации и samplers

CPU geometry применяется в Dataset, GPU transforms — после переноса batch и перед моделью. Встроенный `SampleTransform`, Albumentations и Kornia синхронно преобразуют image и связанные input/target/validity masks; masks используют nearest interpolation. Normalization задаётся один раз. Text adapters: `ByteTokenizer`, `HFTokenizer` с revision/max length.

`sampler=field_weighted` балансирует произвольное `field`, включая `manifest_name`, label или metadata. Есть sample-weighted sampling и `sampler=pk`; PK гарантирует P групп × K примеров **на каждом rank**. Replacement, missing/insufficient policy и epoch seed задаются явно. Online train metrics описывают все sampled observations, включая реальные повторы.

MixUp/CutMix подключаются как `gpu_transforms._target_: trainer_project.transforms.MixUp`/`CutMix` с `alpha`, `target_classes`, `image_keys`, `seed`. Они требуют полностью валидных и разрешённых categorical labels для всех объявленных target fields. CE принимает soft targets; несовместимые losses и SSL views отклоняются. Hard-label F1 для mixed observations не публикуется.

TTA в evaluation:

```yaml
tta:
  enabled: true
  transforms: [identity, hflip, vflip]
  aggregation: mean_probabilities
```

Геометрия dense outputs отменяется до агрегации; для binary/multilabel укажите соответствующий `target_mode`.

## Checkpoints, resume и reproduce

Каждый run получает новый ID. Рядом с полным `.ckpt` лежит fully resolved `.yaml`; конфиг, hash, class maps, data identity, code identity и weights provenance также встроены в checkpoint.

```bash
uv run --no-sync trainer resume --checkpoint runs/RUN/checkpoints/last.ckpt --strict
uv run --no-sync trainer reproduce --config runs/RUN/checkpoints/last.yaml
uv run --no-sync trainer branch --checkpoint runs/RUN/checkpoints/best-file.ckpt optimizer.lr=0.0001
```

Путь лучшего checkpoint находится в `result.json` и `checkpoints/registry.json`; имена содержат epoch/step. `last.ckpt` предназначен для продолжения. `--stop-after-epoch 1` останавливает запуск после первой эпохи без изменения scheduler horizon.

Strict resume проверяет завершённую границу эпохи, рецепт, topology, batch/accumulation, scheduler horizon, версии окружения, код, manifests, content hashes изображений/масок и порядок классов. RNG Python/NumPy/PyTorch/CUDA собираются по ranks, sampler и augmentation state восстанавливаются. Для workers используется `persistent_workers=false`; эпоха устанавливается до создания iterator. Mid-epoch checkpoint не объявляется пригодным для strict epoch resume.

Reproduce использует сохранённый рецепт и `initialization.pt`, содержащий начальные model/objective weights и RNG. Инициализация наследуется при resume, hashes проверяются. Если переносится только checkpoint без исходной инициализации, inference/resume доступны, а exact reproduce сообщает об отсутствии необходимого артефакта. `--path-map paths.yaml` у resume/reproduce переназначает корни данных; содержимое остаётся проверяемым.

Branch/finetune создаёт новый optimizer/scheduler и записывает parent checkpoint/hash/diff overrides. При изменении числа классов classifier head инициализируется заново; несовместимые размеры backbone отклоняются. `export --checkpoint ... --output runs/weights.pt` создаёт явно помеченный weights-only artifact, непригодный для resume.

**Файлы не удаляются.** Top-K registry определяет лучшие checkpoints, вытесненные файлы сохраняются. Предыдущие версии `last.ckpt` и staging files также сохраняются. Новые epoch files публикуются атомарно; обновление совместимого имени `last.ckpt` выполняется записью с сохранением предыдущей копии. При аварии во время такой записи доступен предыдущий полный checkpoint. Политика сохранения имеет приоритет над освобождением дискового пространства.

Run содержит source/resolved configs и overrides, class maps/schema, preprocessing, копии manifests, content identities, initialization/weights lock, `uv.lock`, environment, snapshot исходников, metrics JSONL, TensorBoard и prediction/report artifacts. Встроенных исходников Git нет: код зафиксирован копиями и SHA-256.

## Оценка и отчёты

```bash
uv run --no-sync trainer evaluate --checkpoint runs/RUN/checkpoints/last.ckpt
uv run --no-sync trainer evaluate --checkpoint runs/RUN/checkpoints/last.ckpt --eval-config evaluation.yaml
uv run --no-sync trainer predict --checkpoint runs/RUN/checkpoints/last.ckpt --manifest examples/demo_data/val_a.csv
uv run --no-sync trainer extract_embeddings --checkpoint runs/RUN/checkpoints/last.ckpt
uv run --no-sync trainer report --predictions runs/EVAL/predictions.jsonl
```

Standalone evaluation по умолчанию работает на CPU; в `evaluation.yaml` можно указать `device: cuda:0`, новые `data.val`/`data.test`, `evaluation` и TTA. Модель, class mapping и preprocessing восстанавливаются из checkpoint. Подмена class order запрещена.

`val/<name>/...`, `val/pooled/...`, `test/<name>/...` и `test/pooled/...` независимы. Pooled F1 строится из общей confusion matrix. Для всех categorical targets сохраняются validity, coverage, per-class precision/recall/F1/support, accuracy, AUC и calibration metrics с N/A для неопределённых случаев. Segmentation использует суммарные pixel counts для Dice/IoU. Technical DDP padding исключается; реальные повторения между manifests регулируются `evaluation.pooled_policy: observations|deduplicate`.

По умолчанию test observers оцениваются при каждом validation event. Частота задаётся `evaluation.test_during_fit.every_n_validation_events`; пропущенная оценка не публикует stale metrics и не продвигает monitor-driven state. Выбор модели по test monitor отмечается в provenance как tuning.

HTML/JSON/CSV отчёт перестраивается из JSONL predictions без модели и GPU. В predictions входят IDs, происхождение/роль, все targets/validity, logits/probabilities/embeddings или dense masks, class maps, checkpoint identity и evaluation protocol hash.

## SSL kNN и linear probe

Reference задаётся явно; базовый вариант — labeled train rows. Метрики рассчитываются только на val. Validation не делится на folds, его labels не участвуют в обучении probe. MAE features извлекаются без masking, encoder/BN заморожены, probe использует отдельный optimizer и RNG.

```bash
uv run --no-sync trainer knn --checkpoint runs/SSL/checkpoints/last.ckpt --eval-config examples/ssl_evaluation.yaml
uv run --no-sync trainer linear_probe --checkpoint runs/SSL/checkpoints/last.ckpt --eval-config examples/ssl_evaluation.yaml
```

Тот же `ssl_evaluation` блок можно включить в training config. Доступен явно названный `knn.protocol: val_leave_one_out`; self/group исключаются. Полноценный linear probe требует независимого labeled reference. Допускается отдельный reference manifest. Classification/segmentation checkpoints для этих команд отклоняются.

## Ансамбли, CV и OOF

```bash
uv run --no-sync trainer ensemble --predictions runs/A/predictions.jsonl runs/B/predictions.jsonl --weights 1 2
uv run --no-sync trainer cv --config prepared_folds.yaml
uv run --no-sync trainer oof --predictions runs/CV/oof_predictions.jsonl --membership runs/CV/training_membership.json
```

Ensemble выравнивает identity и **имена классов**, проверяет полноту samples и согласованность targets. `mean_probabilities` и `mean_logits` имеют разные явно выбранные режимы. Каждый checkpoint предварительно оценивается со своим preprocessing.

CV config содержит `folds: [examples/fold_0.yaml, examples/fold_1.yaml]`, optional `identity_key`, `require_image_independence` и `output_dir`. Runner обучает предоставленные splits, оценивает лучшие checkpoints, проверяет label/image leakage, сохраняет OOF predictions, pooled metrics, mean/std folds и отчёт. Усреднение всех моделей на training rows не называется OOF. SSL exposure отмечается отдельно как transductive representation learning. Calibration доступна через `trainer_project.evaluation.TemperatureCalibrator` с обязательным указанием validation provenance.

## Расширение

Model возвращает словарь named outputs. Objective — зарегистрированный `nn.Module`:

```python
from trainer_project.objectives import Objective, ObjectiveResult

class CustomObjective(Objective):
    def forward(self, outputs, batch, context=None):
        # batch: inputs/input_present/views/targets/target_valid/
        # supervision_allowed/sample_id/manifest_name/metadata
        # Для supervised-компонента используйте validity & permission mask.
        loss = ...
        return ObjectiveResult(loss, components={"custom": loss.detach()}, counts={"custom": ...})
```

Model получает inputs/views без истинных labels. Trainable parameters objective входят в optimizer/DDP/checkpoint. Для DDP masked loss используется `global_masked_mean`, учитывающий глобальный denominator и последующее усреднение gradients. Custom model/objective/dataset/sampler/augmentation/evaluator/callback задаются через импортируемый `_target_`; optimizer/scheduler создаются после model parameters. Примеры контрактов и инвариантов находятся в `tests/`.

## Проверка

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check --config pyproject.toml src tests examples
uv run --no-sync python examples/verify_gpu.py
```

Последняя команда требует две CUDA GPU, сохраняет все логи и проверяет exact DDP epoch resume плюс AMP/accumulation. Pytest также проверяет пропущенный AMP step и BYOL EMA при наличии CUDA. Внешние адаптеры проверены offline; сетевое скачивание конкретных пользовательских pretrained weights не входит в smoke tests.

Multi-node/FSDP, strict mid-epoch resume, hierarchical classification и model soup остаются расширениями, как предусмотрено ТЗ.

Основа lifecycle следует официальным контрактам [Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html), [Lightning checkpointing](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html), [Hydra instantiate](https://hydra.cc/docs/advanced/instantiate_objects/overview/) и [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
