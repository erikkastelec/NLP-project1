import pickle
import os


def print_best(all):
    print(f'\nRelationships: \n')
    sort_by_rel_accf = sorted(all, key=lambda x: x[0]['relationships']['accuracy_fonly'], reverse=True)
    print(f'Best accuracy_fonly: {sort_by_rel_accf[0][2]} - {sort_by_rel_accf[0][0]["relationships"]["accuracy_fonly"]}')
    sort_by_rel_acc = sorted(all, key=lambda x: x[0]['relationships']['accuracy'], reverse=True)
    print(f'Best accuracy: {sort_by_rel_acc[0][2]} - {sort_by_rel_acc[0][0]["relationships"]["accuracy"]}')
    sort_by_rel_f1 = sorted(all, key=lambda x: x[0]['relationships']['f1'], reverse=True)
    print(f'Best f1: {sort_by_rel_f1[0][2]} - {sort_by_rel_f1[0][0]["relationships"]["f1"]}')
    sort_by_rel_precision = sorted(all, key=lambda x: x[0]['relationships']['precision'], reverse=True)
    print(f'Best precision: {sort_by_rel_precision[0][2]} - {sort_by_rel_precision[0][0]["relationships"]["precision"]}')
    sort_by_rel_recall = sorted(all, key=lambda x: x[0]['relationships']['recall'], reverse=True)
    print(f'Best recall: {sort_by_rel_recall[0][2]} - {sort_by_rel_recall[0][0]["relationships"]["recall"]}')

    all_pa = [a for a in all if len(a[0]['prot/ant']) > 0]
    print(f'\nProt/Ant: \n')
    sort_by_acc = sorted(all_pa, key=lambda x: x[0]['prot/ant']['accuracy'], reverse=True)
    print(f'Best accuracy: {sort_by_acc[0][2]} - {sort_by_acc[0][0]["prot/ant"]["accuracy"]}')
    sort_by_f1 = sorted(all_pa, key=lambda x: x[0]['prot/ant']['f1'], reverse=True)
    print(f'Best f1: {sort_by_f1[0][2]} - {sort_by_f1[0][0]["prot/ant"]["f1"]}')
    sort_by_precision = sorted(all_pa, key=lambda x: x[0]['prot/ant']['precision'], reverse=True)
    print(f'Best precision: {sort_by_precision[0][2]} - {sort_by_precision[0][0]["prot/ant"]["precision"]}')
    sort_by_recall = sorted(all_pa, key=lambda x: x[0]['prot/ant']['recall'], reverse=True)
    print(f'Best recall: {sort_by_recall[0][2]} - {sort_by_recall[0][0]["prot/ant"]["recall"]}')

def find(all, name):
    for a in all:
        if a[2] == name:
            return a[0]

if __name__ == '__main__':
    all = []
    files = os.listdir('pickles')
    paths = [f.split('.')[0] for f in files if f.find('ner.') != -1]
    paths = [('_').join(f.split('_')[:-1]) for f in paths]

    for path in paths:
        with open('pickles/' + path + '_books.pickle', 'rb') as f:
            books = pickle.load(f)
        with open('pickles/' + path + '_corpus_eval.pickle', 'rb') as f:
            corpus_eval = pickle.load(f)
        all.append((corpus_eval, books, path))

    print('\n\nGold+Normal:')
    print_best(all)

    print('\n\nNormal:')
    alls = [a for a in all if a[2].find('gold') == -1]
    print_best(alls)

    print('\n\nGold:')
    all_gold = [a for a in all if a[2].find('gold') != -1]

    print_best(all_gold)

    pass