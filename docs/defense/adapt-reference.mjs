// Adapt the user-supplied 11-slide pitch in place, preserving its theme/layouts.
// Usage: node docs/defense/adapt-reference.mjs <reference.pptx>
// Requires the bundled runtime env paths described in this directory's README.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(here,'../..');
const source=process.argv[2];
const modules=process.env.ARTIFACT_NODE_MODULES;
const skill=process.env.PRESENTATIONS_SKILL;
if(!source||!modules||!skill||!process.env.RUNTIME_PYTHON) throw new Error('Supply reference PPTX and bundled runtime paths');
process.env.RUNTIME_NODE_MODULES ||= modules;
const build=path.resolve(process.env.BUILD_DIR||path.join(root,'data/defense-prep/pitch-reference-v1'));
await fs.mkdir(build,{recursive:true});
const sourceBytes=await fs.readFile(source);
const sourceSha=createHash('sha256').update(sourceBytes).digest('hex');
const snapshot=path.join(build,'reference.pptx');
try { await fs.writeFile(snapshot,sourceBytes,{flag:'wx'}); }
catch(error) {
 if(error.code!=='EEXIST'||createHash('sha256').update(await fs.readFile(snapshot)).digest('hex')!==sourceSha) throw error;
}
const {PresentationFile,FileBlob}=await import(pathToFileURL(path.join(modules,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const p=await PresentationFile.importPptx(await FileBlob.load(snapshot));
if(p.slides.items.length!==11) throw new Error('Expected the supplied 11-slide reference');
const records=(await p.inspect({kind:'textbox,shape,image',maxChars:1000000})).ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x));
function find(n,name,kind='textbox') {
 const r=records.find(x=>x.slideIndex===n-1&&x.kind===kind&&x.name===name);
 if(!r) throw new Error(`Missing source ${n}/${name}/${kind}`);
 return r;
}
function set(n,index,value,style={}) {
 const r=find(n,`Text ${index}`);
 const shape=p.resolve(r.id);
 shape.text=value;
 shape.text.style={...r.style,fontFamily:'Arial',typeface:'Arial',...style};
}
async function replaceImage(n,name,asset,alt) {
 const r=find(n,name,'image');
 const im=p.resolve(r.id);
 const bytes=await fs.readFile(path.join(here,'assets',asset));
 const frame=im.frame;
 im.replace({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),
             contentType:'image/png',alt,fit:'contain'});
 im.frame=frame;
}

set(1,6,'Контроль опасных зон, предупреждения и доказательные кадры');
set(2,18,'Детекция людей: контрольный кадр команды');
set(2,19,'Оператор проверяет подтверждённый эпизод по сохранённому кадру.');
set(3,4,'Плавный видеопоток работает независимо от удалённого инференса.');
// The gateway receives JPEG over RabbitMQ. Correct the source's visual order.
set(3,10,'RabbitMQ'); set(3,11,'запрос / ответ\nнезависимые маршруты');
set(3,14,'JPEG Gateway'); set(3,15,'JPEG в тензор\nв облачном контуре');
set(3,29,'Демонстрация: обнаружение, Warns и Backlog. Видео продолжает работать при задержках анализа.');
set(4,3,'Пример 1 / настройка опасной зоны');
set(4,4,'Полигон задаётся в конфигурации. Правило использует точку у ног человека.');
set(4,7,'Опасный участок\nзадаётся полигоном\nв конфигурации камеры');
set(4,9,'Нормализованные координаты от 0 до 1.\nВход проверяется по нижней центральной точке рамки.');
set(4,11,'ЭКРАН ПРИЛОЖЕНИЯ / DEMO');
await replaceImage(4,'Image 0','stream.png','Текущий интерфейс SafeSight в синтетическом DEMO, без нейросети и TPU');
set(5,21,'Для зон нужны три последовательных наблюдения. Частота preview не заменяет подтверждение.');
set(6,5,'Иллюстрация сцены');
set(6,6,'Текущий журнал / DEMO');
set(6,7,'Слева иллюстрация из макета. Справа реальный интерфейс DEMO с отдельным журналом и JPEG.');
await replaceImage(6,'Image 1','backlog.png','Текущий Backlog SafeSight: DEMO, источник, JPEG и операторская отметка');
set(7,4,'Общий журнал объединяет людей, контроль зон и экспериментальные сценарии SafetyVision.');
set(7,6,'ОСНОВНОЙ СЦЕНАРИЙ / ЛЮДИ И ЗОНЫ');
set(7,9,'Аппаратный путь через LinQ TPU.\nКонтроль зон, предупреждения и архив evidence.');
set(7,11,'ДОПОЛНИТЕЛЬНЫЙ СЦЕНАРИЙ / ПИЛОТ');
set(7,14,'Сопоставление СИЗ с человеком.\nПодтверждение по нескольким наблюдениям.');
set(7,15,'Контроль отсутствия каски дорабатывается. Перед внедрением нужна проверка на видео площадки.');
set(8,3,'Подтверждённые результаты');
set(8,4,'Локальный ONNX: диагностическая выборка из 20 интернет-снимков.');
set(8,6,'ЛЮДИ И ПАДЕНИЯ / LOCAL ONNX CPU');
set(8,7,'92,6'); set(8,8,'%'); set(8,9,'Precision людей');
set(8,11,'80'); set(8,12,'%'); set(8,13,'Recall падений');
set(8,15,'IoU ≥ 0,5 / 20 снимков');
set(8,17,'НАЛИЧИЕ КАСКИ / HARDHAT');
set(8,18,'100% / 80%');
set(8,19,'Precision / recall, 10 кадров');
set(8,21,'ПРОВЕРКА ПРИЛОЖЕНИЯ');
set(8,22,'112 + 9');
set(8,23,'Python + JavaScript тесты.\nCI на Windows, Linux и macOS.');
set(8,24,'Малая выборка, LOCAL ONNX CPU. Контроль отсутствия каски требует доработки. Это не оценка TPU.');
set(9,4,'Оператор получает событие с кадром, временем и результатом проверки.');
set(9,8,'Детекция людей и проверка настроенной зоны');
set(9,18,'Демосценарий: зона, Warns и Backlog с доказательным кадром');
set(9,19,'Следующий этап: валидация SafetyVision на видео площадки и парный ONNX/TPU тест.');
set(10,3,'Резерв / методика и границы измерений');
set(10,4,'Сырые рамки, IoU ≥ 0,5. Малая диагностика, не независимый промышленный benchmark.');
set(10,6,'Люди / YOLOv8n'); set(10,7,'F1 86,2%');
set(10,8,'P 92,6%, R 80,6%. TP/FP/FN: 25/2/6.\n19 кадров с проверенной разметкой.');
set(10,10,'Падения / SV'); set(10,11,'F1 88,9%');
set(10,12,'P 100%, R 80%. TP/FP/FN: 4/0/1.\n10 кадров, 5 положительных случаев.');
set(10,14,'Наличие каски'); set(10,15,'P 100%');
set(10,16,'R 80%, F1 88,9%. TP/FP/FN: 8/0/2.\nКласс Hardhat, 10 кадров.');
set(10,18,'Отсутствие каски'); set(10,19,'1/5 сцен');
set(10,20,'Наличие класса на фото. Строгие рамки:\nTP/FP/FN: 0/1/12. Требуется доработка.');
set(10,21,'Источник: docs/INTERNET20_RESULTS.md. Результаты ONNX не подменяют аппаратный тест и видеотревоги.');
set(11,4,'Инструкции запуска, доступность аппаратного контура и границы пилота.');
set(11,7,'Windows Web, VPN и два Gateway.\nМаршруты берём из GATEWAY READY.');
set(11,10,'SQLite, JPEG и дисковая очередь записи.\nОтдельная история операторских решений.');
set(11,13,'Видео независимо от инференса.\nDEMO явно помечен при отсутствии TPU.');
set(11,16,'Каски и падения требуют видео-теста.\nНет промышленной сертификации.');
set(11,17,'Запуск и проверка: docs/RUNBOOK.md и docs/READINESS.md. Перед показом проверить режим и AMQP.');

const metrics='Источники: docs/INTERNET20_RESULTS.md, docs/reports/safetyvision_internet20.json. LOCAL ONNX CPU. 20 одинаковых JPEG для обеих моделей, person score на 19 после проверки разметки, один неполно размеченный crowd исключён. PPE и fall по 10 кадров. Каска Hardhat означает наличие СИЗ, не надёжность детекции его отсутствия. No-helmet: presence 1/5 положительных, строгие рамки TP/FP/FN 0/1/12. Пересечение с обучением неизвестно. Новых аппаратных raw/quantization метрик нет. Полные показатели сохранены в резерве и отчёте.';
for(let i=0;i<11;i++) p.slides.items[i].speakerNotes.text += '\n\nИсточник оформления: предоставленная пользователем презентация. Изображения из макета служат иллюстрациями. Текущие скриншоты приложения помечены DEMO. Исходная презентация не изменена.';
p.slides.items[3].speakerNotes.text+='\nТекущая версия задаёт зоны через config/cameras.json. Графическое рисование полигона из исходного макета не заявляется как готовая функция.';
p.slides.items[7].speakerNotes.text+='\n'+metrics;
p.slides.items[9].speakerNotes.text+='\n'+metrics;
const timings=[12,15,18,20,15,20,15,20,15,0,0];
for(let i=0;i<11;i++) p.slides.items[i].speakerNotes.text+=`\nРекомендуемое время: ${timings[i]} с. Между слайдами 6 и 7 показать живое DEMO 30 с. Первые 9 слайдов и показ: 180 с. Слайды 10–11 резервные.`;
for (const [i,s] of p.slides.items.entries()) {
 const png=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
 await fs.writeFile(path.join(build,`layout-${i+1}.json`),await(await s.export({format:'layout'})).text());
}
const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,
 finalPath:path.join(here,process.env.DECK_NAME||'SafeSight-defense-3min.pptx'),
 pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
 explicitTotalSlideCount:11,
 fontPolicy:{basis:'reference',families:['Arial'],referencePath:snapshot,referenceSha256:sourceSha},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify({sourceSha,slides:11,result}));
