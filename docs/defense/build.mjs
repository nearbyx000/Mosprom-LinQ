// Editable presentation source. Requires the Codex bundled @oai/artifact-tool.
// Set ARTIFACT_NODE_MODULES, PRESENTATIONS_SKILL, RUNTIME_PYTHON and BUILD_DIR.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(here,'../..');
const build=path.resolve(process.env.BUILD_DIR || path.join(root,'data/defense-prep/deck-build'));
const skill=process.env.PRESENTATIONS_SKILL;
const modules=process.env.ARTIFACT_NODE_MODULES;
if (!skill || !modules || !process.env.RUNTIME_PYTHON) throw new Error('Set the bundled runtime paths documented in README.md');
process.env.RUNTIME_NODE_MODULES ||= modules;
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(modules,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const {resolvePresentationFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const font=resolvePresentationFont();
const p=Presentation.create({slideSize:{width:1280,height:720}});
const colors={bg:'#0B1119',text:'#E4EBF3',muted:'#A4B4C6',green:'#7EE8BA',amber:'#FFC56A',line:'#344659'};
function text(s,value,x,y,w,h,size=28,color=colors.text,bold=false) {
  const shape=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;
  shape.text.style={fontFamily:font,typeface:font,fontSize:size,color,bold};
  return shape;
}
function slide(title,notes) {
  const s=p.slides.add();s.background.fill=colors.bg;
  text(s,title,64,50,1152,108,46,colors.text,true);
  s.speakerNotes.text=notes;
  return s;
}
async function image(s,name,x,y,w,h) {
  const bytes=await fs.readFile(path.join(here,'assets',name));
  s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:'SafeSight: синтетическое DEMO без нейросети и TPU',fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
function table(s,values,widths,x=64,y=182,h=320,size=25) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:widths.reduce((a,b)=>a+b),height:h,columnWidths:widths,values});
  t.borders.assign({fill:colors.line,width:1,style:'solid'});
  for(let r=0;r<values.length;r++) for(let c=0;c<values[r].length;c++) {
    const cell=t.getCell(r,c);cell.fill=r===0?'#18352D':colors.bg;
    cell.text.style={fontFamily:font,typeface:font,fontSize:size,color:r===0?colors.green:colors.text,bold:r===0};
  }
  return t;
}

let s=slide('SafeSight / LinQ','Источник: README.md, docs/BUSINESS_CASE.md. Проверяемый пилот одной камеры, версия 0.3. Не заявлять production/SLA или гарантированное предотвращение травм.');
text(s,'Операторский контроль\nпромышленной безопасности',64,215,1080,190,58,colors.text,true);
text(s,'Видео · опасные зоны · журнал с доказательными кадрами',64,470,1100,90,30,colors.green);
text(s,'Пилот одной камеры  /  25 октября 2026',64,620,1100,50,24,colors.muted);

s=slide('Рабочее место оператора','Источник: safesight/static/, docs/DEFENSE.md. Скриншот собственного приложения в DEMO, не промышленная камера. В demo — цветовые фигуры и 2 Гц анализа; в LinQ — удалённый аппаратный путь. FPS preview не равен FPS инференса.');
text(s,'Непрерывное видео\n\nЗоны и свежие рамки\n\nWarns: события сейчас\n\nBacklog: история',64,190,375,365,29);
await image(s,'stream.png',460,172,756,430);
text(s,'На экране — DEMO. Нейросеть и TPU в этом сценарии не используются.',64,636,1150,58,23,colors.amber);

s=slide('Проверка и разбор события','Источник: safesight/evidence.py, storage.py, exports.py, docs/RUNBOOK.md. JPEG + SHA256, timestamp захвата, frame/generation/source. Review хранится append-only, отдельно от исходного события; ручной review не является автоматическим измерением recall.');
text(s,'Сохранённый JPEG\nи источник события\n\nОтметка оператора\nс историей решений\n\nФильтры и CSV',64,190,360,390,29);
await image(s,'backlog.png',440,172,776,440);
text(s,'При временном отказе SQLite событие восстанавливается из дисковой очереди.',64,636,1150,58,23,colors.green);

s=slide('Надёжность контура','Источник: docs/ARCHITECTURE.md; safesight/pipeline.py, dual_gateway.py, safety.py, logic.py, evidence.py. Метрики/автотесты не доказывают работу удалённого TPU в текущей сети.');
table(s,[['Условие','Поведение системы'],['TPU или SQLite отвечает медленно','Видео продолжает обновляться'],['Ответ относится к старому кадру','Результат отбрасывается'],['SafetyVision не ответила','СИЗ = неизвестно; YOLOv8n продолжает работу'],['Каска и no-helmet противоречат друг другу','Сопоставление с человеком; неоднозначность не даёт тревогу'],['Одиночная подозрительная детекция','Повторное подтверждение по наблюдениям']],[420,732],64,180,414,25);
text(s,'Рабочие TPU-программы и routes сохранены. Локальной подмены в LinQ нет.',64,636,1150,58,23,colors.green);

s=slide('Диагностика на 20 изображениях','Источник: docs/INTERNET20_RESULTS.md; docs/reports/safetyvision_internet20.json. Интернет-источники: https://docs.ultralytics.com/datasets/detect/construction-ppe ; https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset . На всех 20 выполнены обе ONNX модели; person score по 19 после предынференсной проверки GT, один неполно размеченный crowd исключён. Данные не независимый benchmark; пересечение train неизвестно.');
text(s,'LOCAL ONNX CPU · строгие рамки IoU ≥ 0.50 · не TPU',64,145,1150,50,26,colors.green);
table(s,[['Модель / класс','Кадры','TP / FP / FN','P','R','F1'],['YOLOv8n / человек','19','25 / 2 / 6','0.926','0.806','0.862'],['SafetyVision / каска','10','8 / 0 / 2','1.000','0.800','0.889'],['SafetyVision / без каски','10','0 / 1 / 12','0.000','0.000','0.000'],['SafetyVision / падение','10','4 / 0 / 1','1.000','0.800','0.889']],[395,100,222,145,145,145],64,213,310,24);
text(s,'По наличию класса на изображении: без каски 1 из 5; падение 4 из 5.\nНа пяти отрицательных кадрах каждого сценария FP = 0. Это не видеотревоги.',64,551,1150,100,25,colors.amber);
text(s,'Малая диагностическая выборка. Новых аппаратных метрик нет: AMQP handshake не прошёл.',64,666,1150,40,20,colors.muted);

s=slide('Проверка исправлений','Источник: docs/reports/safetyvision_after_rules.json, tests/fixtures/safetyvision_baseline_events.json; 16 синтетических сценариев, 8 истинных эпизодов в 7 положительных случаях. Python 112 passed в рабочем и чистом окружениях, JS 9 passed. Это регрессия логики, не neural accuracy. Дополнительно проверена запись одного события при переполнении очереди: docs/reports/storage_backpressure.json.');
table(s,[['Синтетические правила','До','После'],['TP / FP / FN','6 / 7 / 2','8 / 0 / 0'],['Precision / Recall','0.462 / 0.750','1.000 / 1.000'],['F1','0.571','1.000']],[570,290,292],64,186,280,29);
text(s,'112 Python-тестов + 9 JavaScript-тестов\nЗапуск без моделей и SDK; браузерная проверка пульта',64,510,1150,95,30,colors.green,true);
text(s,'На 20 реальных снимках агрегатные метрики исходной модели не улучшились.\nОтдельный helmet-кандидат отклонён: PPE F1 0.378 против 0.516 у SafetyVision.',64,631,1150,72,22,colors.amber);

s=slide('План пилота на площадке','Источник: docs/BUSINESS_CASE.md, docs/MODEL_EXPERIMENTS.md, docs/SAFETYVISION_EVALUATION.md. Экономический эффект не измерен; нет обещаний ROI/SLA, HA/RBAC/SSO и сертификации. Нужны согласованная видеоразметка, права на веса/данные и независимая оценка.');
table(s,[['Этап','Результат для решения о внедрении'],['1. Одна камера, зоны и оператор','Согласованные сцены, регламент реакции и журнал проверок'],['2. СИЗ и падения на видео цеха','Recall по контрольным эпизодам; ложные тревоги на камеро-час'],['3. ONNX ↔ LinQ на одних входах','Парные raw-выходы; оценка квантования и калибровки'],['4. Решение о расширении','Критерии качества, стоимость поддержки, доступ и хранение данных']],[430,722],64,190,330,27);
text(s,'Приоритет: снизить пропуски отсутствия каски.\nПилот СИЗ требует улучшения модели и независимой повторной проверки.',64,561,1150,100,30,colors.amber,true);

await fs.mkdir(build,{recursive:true});
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
for (const [i,slide] of p.slides.items.entries()) {
  const png=await p.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
const output=path.join(here,process.env.DECK_NAME || 'SafeSight-defense-v2.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:output,
  pythonExecutable:process.env.RUNTIME_PYTHON,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[4,5,6,7].flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:7,requiredNativeTableOwnerSlides:[4,5,6,7],
  fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
  receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify({font,output,result}));
