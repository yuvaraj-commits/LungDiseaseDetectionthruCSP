import os
import pandas
def remove(weights=False, diff=4, res_path=None, high=92, low=90, weights_c=False):
    if res_path:
        already_removed = 0
        removed = 0
        df = pandas.read_csv(res_path)
        df = df.loc[(df['train_accuracies']-df['val_accuracies'] > diff) | (df['train_accuracies']<df['val_accuracies'])]
        df = df.loc[(df['train_accuracies']<high) | (df['val_accuracies']<low) ]
        remove_paths = df['save_model_paths'].unique()
        for remove_path in remove_paths:
            if os.path.exists(remove_path):
                if '.pt' in remove_path:
                    os.remove(remove_path)
                    removed += 1
            else:
                already_removed += 1
        print(res_path, 'filename')
        print(already_removed, 'weights already_removed')
        print(removed, 'weights removed')
        return
    weights = False
    result_path = os.path.join(os.getcwd(), 'results')
    valid_weights_paths = []
    for root, dir, files in os.walk(result_path):
        print(root)
        already_removed = 0
        removed = 0
        for filename in files:
            if 'results.csv' in filename:
                pass
            elif 'res_temp.txt' in filename:
                if not weights_c:
                    print(filename, 'skip')
                    continue
            else:
                print(filename, 'skip')
                continue
                
            path = os.path.join(root, filename)
            df = pandas.read_csv(path)
            valid_weights_paths += list(df['save_model_paths'].unique())
            df = df.loc[(df['train_accuracies']-df['val_accuracies'] > diff) | (df['train_accuracies']<df['val_accuracies'])]
            df = df.loc[(df['train_accuracies']<high) | (df['val_accuracies']<low) ]
            remove_paths = df['save_model_paths'].unique()
            for remove_path in remove_paths:
                if os.path.exists(remove_path):
                    if '.pt' in remove_path:
                        os.remove(remove_path)
                        removed += 1
                else:
                    already_removed += 1
            print(filename, 'filename')
            print(already_removed, 'weights already_removed')
            print(removed, 'weights removed')
        print('*'*30)
#     if weights:
#         result_path = os.path.join(os.getcwd(), 'weights')
#         removed = 0
#         for root, dir, files in os.walk(result_path):
#             print(root)
#             already_removed = 0
#             removed = 0
#             for filename in files:
#                 if '.pt' in filename:
#                     if filename not in valid_weights_paths:
#                         os.remove(os.path.join(root, filename))
#                         removed += 1
#             print(removed)
#             print('*'*20)
        