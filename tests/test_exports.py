import csv
import io

from safesight.exports import event_csv


def test_csv_preserves_cyrillic_and_escapes_formulas():
    result = event_csv([{'id':1,'camera_id':'Камера','zone_id':'=HYPERLINK("unsafe")',
                         'violation_type':'restricted_zone','source':'demo','confidence':.9}])
    assert result.startswith(b'\xef\xbb\xbf')
    rows = list(csv.reader(io.StringIO(result.decode('utf-8-sig'))))
    assert rows[1][2] == 'Камера'
    assert rows[1][3].startswith("'=")
    assert rows[1][7] == 'demo'
