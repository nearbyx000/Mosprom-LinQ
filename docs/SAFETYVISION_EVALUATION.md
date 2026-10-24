# SafetyVision: запуск и воспроизводимая оценка

Все команды выполняются из корня репозитория. Отчёты и каталоги экспериментов
должны иметь новые имена: инструменты отказываются перезаписывать результат.
Для Windows замените `python` на `.\.venv\Scripts\python.exe`.

## Web и два аппаратных Gateway

Работающий YOLOv8n Gateway оставьте на его прежнем route. Для SafetyVision
используйте отдельный каталог на сервере с обновлёнными `gateway.py`,
`decoder.py`, `preprocessing.py` из `gateways/safetyvision`.
В окружении этого Gateway нужны `numpy`, `Pillow`, `opencv-python-headless`,
`pika` и официальный `tpu_pycloud_format`. OpenCV нужен именно обновлённому SV.
Сначала проверьте его наличие: `python -c "import cv2; print(cv2.__version__)"`.

```bash
python -u gateway.py --program /auto/cache/YOUR_LOGIN/safetyvision_v2_640.tpu --timeout 12
```

Это использует существующую программу и автоматически создаёт новый уникальный
`safesight.jpeg.sv.*` route. Процесс не компилирует и не загружает модель.
Если AMQP недоступен, исправьте соединение/VPN/учётные данные в этой среде;
не переходите незаметно на локальную модель. После запуска используйте оба
`GATEWAY READY` route в локальном Web:

```powershell
$env:LINQ_JPEG_ROUTE='safesight.jpeg.CURRENT_PRIMARY_ROUTE'
$env:LINQ_SV_ROUTE='safesight.jpeg.sv.CURRENT_SECONDARY_ROUTE'
$env:SAFESIGHT_SV_THRESHOLDS='{"no_helmet":0.45,"fall_detected":0.45,"hardhat":0.40}'
python run.py --mode linq --source 0
```

Либо заполните копию `.env.example` и добавьте `--env-file .env`.
Состояние SV видно в `/api/status`: `rabbitmq.sv_enabled`, `sv_joined`,
`sv_missed`, `sv_error`, отдельно `safetyvision_expected` и `safetyvision_fresh`.
В Warns отсутствие свежих результатов SV показывается отдельно от зон YOLO.

По умолчанию SV отправляет ≤1.5 запроса/с, primary ≤2; одновременно по одному
запросу в каждом клиенте. Видеопоток не ждёт ни одну модель. Объединяются только
одинаковые `(generation, sequence)` и только до `LINQ_MAX_RESULT_AGE` (3 с).
`LINQ_SV_JOIN_GRACE` (0.70 с) задаёт допустимое ожидание SV после ответа primary.
Если `sv_missed` растёт, измерьте SV RTT p95, затем подберите grace в пределах
общего лимита возраста и частоты. Не увеличивайте возраст ради старых тревог.
Между SV-наблюдениями допускается не более `SAFESIGHT_SV_MAX_GAP` (2.5 с).
Окно `SAFESIGHT_SV_CONFIRM_WINDOW=4` требует 3 попаданий; значение 3 включает
строгие три подряд. Primary-only кадр не считается наблюдением СИЗ.

Изменять пороги желательно только в Web: оставьте Gateway confidence 0.40,
чтобы положительная каска сохранялась для проверки противоречий. Порог ниже
Gateway confidence не восстановит отброшенные рамки. При необходимости Gateway
также принимает `--class-thresholds '{"no_helmet":0.45}'`.

## Автотесты и сравнение правил до/после

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
node --test tests/detections.test.cjs tests/operations.test.cjs
python -m scripts.safetyvision_eval replay --cases tests/fixtures/safetyvision_rule_cases.json --output data/eval-rules-new.json
```

Baseline события сохранены в `tests/fixtures/safetyvision_baseline_events.json`;
тест повторно исполняет сохранённый исходный движок `tests/fixtures/baseline_logic.py`,
проверяет его SHA256, события, метрики и хеш входного набора. Для ручного повтора:

```bash
python -m scripts.safetyvision_eval replay --cases tests/fixtures/safetyvision_rule_cases.json --engine-file tests/fixtures/baseline_logic.py --output data/eval-rules-before-new.json
```

Baseline декодера для `internet_eval20 run --legacy-decoder` также включён:
`tests/fixtures/baseline_sv_decoder.py`. Эти два файла сохраняются побайтно через
`.gitattributes`, не импортируются рабочим pipeline и не должны редактироваться.
Хеш JSON-входа канонический; текущие исходники хешируются после нормализации LF.
Импорт произвольного `--engine-file` выполняет Python-код — используйте доверенные
версии. Новый `--output` обязателен; сохранённые отчёты не перезаписываются.

Replay JSON содержит `kind`, список `cases`, у каждого `id`, `expected` и
`frames`. Кадр содержит монотонное время `at` в секундах, `detections` с
`label/confidence/box` и необязательное `safetyvision_observed`.
Рамки `box` — исходные пиксельные `xyxy`; ожидаемые события размечаются в месте
первого подтверждения. Пример полного формата есть в тестовом fixture.
Для реального replay берите person только от primary YOLOv8n и SV того же кадра.
SV Person class в benchmark ниже измеряется как выход исходной SV, но не заменяет
primary person в рабочем Web.

## Размеченный набор

Сохраните **чистые JPEG до рисования рамок** из нужной камеры. Разделите по
камерам/сеансам съёмки на calibration, validation (подбор порогов) и test
(финальный замер). Близкие кадры одного эпизода не должны попадать в разные
части. Нужны каска/нет каски, падение/наклон/сидение, отсутствие людей, перекрытия,
разные размеры, освещение и цвета СИЗ. Разметьте все целевые объекты, включая
явно пустые отрицательные сцены. Сырые видеокадры и персональные данные не
добавляйте автоматически в Git.

Пример `data/test/manifest.json`, пути относительно manifest:

```json
{
  "kind": "site-heldout-test",
  "images": [
    {"id":"cam1-session2-001", "path":"frames/001.jpg", "split":"test",
     "group":"cam1-session2", "tags":["no_helmet"],
     "annotations":[{"label":"no_helmet", "box":[120,30,170,85]}]},
    {"id":"cam1-session3-001", "path":"frames/002.jpg", "split":"test",
     "group":"cam1-session3", "tags":["negative"], "annotations":[]}
  ]
}
```

Разметка должна соответствовать задаче: при оценке всех классов SV размечаются
все классы; при оценке только касок/падений заранее отфильтруйте predictions к
этим классам. Иначе нецелевые выходы справедливо будут учтены как FP.

## ONNX и TPU на одинаковых входах

```bash
python -m pip install -r requirements-safetyvision-eval.txt
python -m scripts.safetyvision_bench infer --backend onnx --onnx safetyvision_v2_640_op11.onnx --manifest data/test/manifest.json --output data/eval-onnx-new
```

Это **LOCAL ONNX CPU / NOT TPU**, только эталон для диагностики. Web не меняет
режим. Каталог содержит JPEG, точный input `.npy`, metadata letterbox, SHA256,
raw выходы и `predictions.json`. По умолчанию используется исправленный OpenCV
preprocessing. Для отдельного исследования старого preprocessing есть
`--preprocess pil`; такой прогон нельзя выдавать за парный quantization test.

В среде с доступом к RabbitMQ и официальным codec выполните:

```bash
python -m scripts.safetyvision_bench infer --backend tpu --inputs data/eval-onnx-new --program /auto/cache/YOUR_LOGIN/safetyvision_v2_640.tpu --timeout 12 --output data/eval-tpu-new
python -m scripts.safetyvision_bench compare --reference data/eval-onnx-new --candidate data/eval-tpu-new --output data/eval-paired-new.json
```

Если codec дан отдельным архивом, добавьте `--codec-path linq_cloud_format.zip`.
При переносе в другую среду копируйте целиком input bundle. TPU режим не
запускает ONNX, не пишет `.tpu` и не меняет настройки. Программу нужно указать
явно. При ошибке задача не отправляется повторно; неполный каталог не имеет
готового `index.json`. После восстановления создайте новый каталог.
Для фиксации происхождения можно передать `--program-sha256`; это заявленный
оператором хеш, автоматической проверки файла на worker нет.

Сравнение проверяет ID, хеш JPEG и input tensor, preprocessing, letterbox и
хеш raw перед вычислениями. Численные ошибки координат и scores считаются
отдельно. `detection_agreement` — совпадение с ONNX, **не precision/recall модели
против ground truth**. Для реальных метрик обоих backend:

```bash
python -m scripts.safetyvision_eval score --manifest data/test/manifest.json --predictions data/eval-onnx-new/predictions.json --output data/quality-onnx-new.json
python -m scripts.safetyvision_eval score --manifest data/test/manifest.json --predictions data/eval-tpu-new/predictions.json --output data/quality-tpu-new.json
```

Метрики: TP/FP/FN, precision/recall/F1 по классам и суммарно, обнаруженные
положительные случаи из N, число ложных тревог и случаев с FP. Сопоставление
один к одному при IoU ≥0.5 (`--iou`); дубликаты — FP; пропущенные ID — ошибка,
не неявный пустой ответ. Деление на ноль даёт `null`, а не выдуманную единицу.
`duration_seconds` в manifest задавайте только для полного размеченного периода
наблюдения: тогда вычисляется false alarms/hour. Для независимых статичных
картинок частота тревог в час не имеет смысла. Оценка тревог из видео выполняется
через replay эпизодов, чтобы не считать каждый кадр новым инцидентом.

Пороги подбирайте на validation с учётом пропусков `no_helmet` и `fall_detected`,
затем один раз замерьте test. При сравнении декодеров сохранённый raw позволяет
пересчитать детекции без новой отправки на общий TPU. Для сравнения старых правил
можно использовать сохранённый `--engine-file`; baseline метрики из отчёта
относятся только к синтетическому fixture, переносить их на test нельзя.

## Калибровка и новая компиляция

Подготовьте отдельный calibration manifest с ≥50 уникальными JPEG, у всех
`split:"calibration"`, непустой `group` и вручную проверенные `tags`.
Обязательное покрытие: `helmet`, `no_helmet`, `fall`, `normal`, `negative`,
`low_light`, `small_person`. Наличие тегов — проверка полноты описания,
не доказательство репрезентативности. Практически полезно несколько сеансов
и десятки кадров каждой сложной ситуации. Исключите соседние дубли.

```bash
python -m scripts.safetyvision_bench calibration --manifest data/calibration/manifest.json --eval-manifest data/test/manifest.json --output data/calibration-candidate-new
```

Проверяются минимальное число, дубли JPEG и тензоров, split, группы и отсутствие
пересечения с test по хешу/группе. Сохраняются точные gateway tensors, которые
должны подаваться в SDK без нового resize или RGB/BGR преобразования.
Случайные изображения/шум не заменяют набор площадки. Существующие evidence с
нарисованными рамками пригодны для smoke-диагностики, но не для этого набора.

В Linux с установленным SDK LinQ и optional eval dependencies:

```bash
python -m scripts.compile_safetyvision --onnx safetyvision_v2_640_op11.onnx --calibration data/calibration-candidate-new --output artifacts/sv-candidate-new
```

Используются API из поставленного `build_yolo_linq.py`: mapping входа,
`calibrate`, `quantize`, batch 1, H1M/O1. Создаются **новые** thresholds,
quantized model, `safetyvision_candidate.tpu` и manifest с хешами.
Этот путь компиляции подготовлен, но в текущей Windows-среде с SDK не выполнялся.
Он запрещает существующий output, calibration <50 и отсутствие проверки test
leakage. Не запускайте поставленный общий `compile.py` без адаптации: его
пример использует случайную калибровку.

Загрузите кандидат на worker под **новым именем** и выполните парный benchmark
и held-out score, указывая этот новый путь. Рабочие `.tpu`, thresholds и route
не заменяйте. Решение о переключении принимается после измерений по каскам,
падениям и ложным тревогам на тех же контрольных случаях.
