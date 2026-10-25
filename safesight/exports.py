"""Operator CSV export with UTF-8 BOM and spreadsheet formula protection."""
import csv
import io


def event_csv(rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['ID','Время UTC','Камера','Зона','Нарушение','Трек','Confidence',
                     'Источник','Кадр','Поколение','JPEG','SHA256 JPEG','Проверка оператора','Комментарий','Проверено UTC'])
    fields = ('id','timestamp','camera_id','zone_id','violation_type','track_id','confidence',
              'source','frame_id','generation','evidence_file','evidence_sha256','review_status','review_note','reviewed_at')
    for row in rows:
        values = []
        for key in fields:
            value = row.get(key,'')
            if value is None:
                value = ''
            if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@','\t','\r','\n')):
                value = "'"+value
            values.append(value)
        writer.writerow(values)
    return b'\xef\xbb\xbf'+stream.getvalue().encode('utf-8')
