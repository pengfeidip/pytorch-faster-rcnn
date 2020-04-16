import easy, os
from multiprocessing import Pool

RUN = 'det_calc_mAP --dt {} > {}'

def get_epochs():
    jsons = [f for f in os.listdir() if f.startswith('epoch') and f.endswith('.bbox.json')]
    return jsons

def calc_map(args):
    easy.run_shell(RUN.format(args[0], args[1]))

if __name__ == '__main__':
    jsons = get_epochs()
    args = []
    for dt in jsons:
        args.append([dt, dt[:-4]+'mAP'])
    p = Pool(20)
    p.map(calc_map, args)
