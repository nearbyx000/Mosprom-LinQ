# Материалы к защите

[Выступление на 3 минуты: 9 основных + 2 резервных слайда](SafeSight-defense-3min.pptx).
[Текст и тайминг живого показа](PITCH_3MIN.md).
Оформление и структура сохранены из предоставленного пользователем образца.
Цифры на слайде 8 взяты из реальных ONNX-отчётов, полная методика и ограничения
вынесены в резерв. Наличие каски (Hardhat) и контроль её отсутствия разделены.

[Подробная версия, 7 слайдов](SafeSight-defense-v2.pptx).
Сценарий выступления и финального чека: [DEFENSE](../DEFENSE.md).

В слайдах — фактические результаты из JSON-отчётов, собственные скриншоты
синтетического DEMO и границы пилота. Адаптация также сохраняет иллюстрации
из предоставленного макета, отмеченные как примеры. Веса моделей не включены.
Текущие ROI-эксперименты подробнее описаны в
[отдельном отчёте](../MODEL_EXPERIMENTS.md).

Исходник — `build.mjs`, таблицы и текст в PPTX редактируемые. Для пересборки
нужен поставляемый с Codex `@oai/artifact-tool` (не зависимость приложения):

```powershell
$env:ARTIFACT_NODE_MODULES = '<Node packages из load_workspace_dependencies>'
$env:RUNTIME_PYTHON = '<Python executable из load_workspace_dependencies>'
$env:PRESENTATIONS_SKILL = '<папка навыка presentations>'
$env:DECK_NAME = 'SafeSight-defense-revised.pptx'
node docs/defense/build.mjs
```

Адаптация предоставленного образца:

```powershell
node docs/defense/adapt-reference.mjs '<путь к SafeSight_Защита_3_минуты_с_демо.pptx>'
```

Нужны те же runtime-переменные. Шаблон SHA256:
`cedb988f034c7a778291ffb2fdaaf967b4445a890bbc79a78bf814e098475cfc`.
Оригинал не изменяется. Новый output задаётся через `DECK_NAME`, отдельный
каталог сборки — `BUILD_DIR`. Рендеры и validation receipt адаптации сохранены
локально в `data/defense-prep/pitch-reference-v1/`. Все 11 слайдов осмотрены.

Готовый файл не перезаписывается. Временные рендеры и validation receipt
остаются в `data/defense-prep/deck-build/`. Макет проверен рендерингом каждого
слайда и структурными валидаторами; открытие в Microsoft PowerPoint отдельно
не выполнялось. Live Artifact Session в этой среде недоступен.
