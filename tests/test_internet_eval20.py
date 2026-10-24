from scripts.internet_eval20 import presence_scores, task_scores


def test_image_presence_is_separate_from_box_localization():
    rows = [{'id':'positive','source':'ppe','truth':{'ppe':[{'label':'no_helmet','box':[0,0,10,10]}]}},
            {'id':'negative','source':'ppe','truth':{'ppe':[]}}]
    predictions = {'positive':[{'class_name':'no_helmet','bbox':[4,4,14,14],'confidence':.8}],
                   'negative':[]}
    boxes = task_scores(rows,predictions,'ppe',{'no_helmet'},'ppe')
    scenes = presence_scores(rows,predictions,'ppe','no_helmet','ppe')
    assert boxes['total']['TP'] == 0
    assert boxes['total']['FP'] == boxes['total']['FN'] == 1
    assert scenes['TP'] == scenes['TN'] == 1
    assert scenes['FP'] == scenes['FN'] == 0


def test_scope_excludes_unannotated_classes_on_other_sources():
    rows = [{'id':'ppe','source':'ppe','truth':{'ppe':[]}},
            {'id':'fall','source':'fall','truth':{'ppe':[]}}]
    predictions = {'ppe':[], 'fall':[{'class_name':'no_helmet','bbox':[0,0,10,10]}]}
    assert presence_scores(rows,predictions,'ppe','no_helmet','ppe')['FP'] == 0
    assert presence_scores(rows,predictions,'ppe','no_helmet')['FP'] == 1
