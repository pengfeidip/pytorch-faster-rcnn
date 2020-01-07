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

def get_iid(s):
    return s.split('|')[0]

def group_by(s):
    res = {}
    for x in s:
        iid = get_iid(x)
        if iid in res:
            res[iid].add(x)
        else:
            res[iid] = set([x])
    return res

if __name__ == '__main__':
    simple = json.load(open('voc2007_trainval.json'))
    voc = json.load(open('voc2007_trainval_no_difficult.json'))
    sim_imgs = set((anno['image_id'] for anno in simple))
    voc_imgs = set((img['id'] for img in voc['images']))

    voc_annos = voc['annotations']
    
    print('are image ids the same:', sim_imgs == voc_imgs)
    print('number of annotations in simple and voc:', len(simple), len(voc_annos))

    print(simple[0], voc_annos[0])

    sim_annos = set((simple_anno_id(anno) for anno in simple))
    voc_annos = set((voc_anno_id(anno) for anno in voc['annotations']))
    print('are annos the same:', sim_annos == voc_annos)
    print(len(sim_annos - voc_annos), len(sim_annos & voc_annos), len(voc_annos-sim_annos))

    iids = set((get_iid(x) for x in sim_annos))
    
    sim_group = group_by(sim_annos)
    voc_group = group_by(voc_annos)
    for iid in sim_group:
        if sim_group[iid] != voc_group[iid]:
            print('**************')
            print(sim_group[iid])
            print(voc_group[iid])
    #print(sim_annos - voc_annos)
