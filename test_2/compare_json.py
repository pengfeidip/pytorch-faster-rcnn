import json


def simple_anno_id(anno):
    bbox = anno['bbox']
    bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]+1), int(bbox[3]+1)]
    return '|'.join([
        str(anno['image_id']),
        str(bbox),
        str(anno['category_id']+1)
    ])

def voc_anno_id(anno):
    return '|'.join([
        str(anno['image_id']),
        str(anno['bbox']),
        str(anno['category_id'])
    ])

if __name__ == '__main__':
    simple = json.load(open('voc2007_test.json'))
    voc = json.load(open('voc2007_test_no_difficult.json'))
    sim_imgs = set((anno['image_id'] for anno in simple))
    voc_imgs = set((img['id'] for img in voc['images']))

    voc_annos = voc['annotations']
    
    print('are image ids the same:', sim_imgs == voc_imgs)
    print('number of annotations in simple and voc:', len(simple), len(voc_annos))

    sim_annos = set((simple_anno_id(anno) for anno in simple))
    voc_annos = set((voc_anno_id(anno) for anno in voc['annotations']))
    print('are annos the same:', sim_annos == voc_annos)
