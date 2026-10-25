# Материалы к защите

[Редактируемая презентация, 7 слайдов](SafeSight-defense-v2.pptx).
Сценарий выступления и финального чека: [DEFENSE](../DEFENSE.md).

В слайдах — фактические результаты из JSON-отчётов, собственные скриншоты
синтетического DEMO и границы пилота. Никаких промышленных кадров или весов
моделей в этом каталоге нет. Текущие ROI-эксперименты подробнее описаны в
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

Готовый файл не перезаписывается. Временные рендеры и validation receipt
остаются в `data/defense-prep/deck-build/`. Макет проверен рендерингом каждого
слайда и структурными валидаторами; открытие в Microsoft PowerPoint отдельно
не выполнялось. Live Artifact Session в этой среде недоступен.
