import pandas as pd
import numpy as np

harmless_vinfo = 'checklist_out/hh-rlhf_regular_vinfo_pvi-pref-t5.csv'

def print_low_pvi_exs(filename, ascending=False):
    df = pd.read_csv(filename, keep_default_na=False)

    cols_to_keep = ['PVI', 'x_raw', 'cond_raw', 'references_labels', 'x_predicted_labels', 'cond_predicted_labels']

    # get ones with low pvi when ascending=True; otherwise get ones with high pvi
    df = df.sort_values(by=['PVI'], ascending=ascending)
    # df = df[df['cond_raw'] != 0.0]
    df = df[df['x_raw'] != '']
    sub_df = df[cols_to_keep].head(50)
    # print(sub_df)

    pvis = []
    for id, row in sub_df.iterrows():
        # print(row['x_raw'])

        if not 'align' in filename:
            context, res = row['x_raw'].split('Response A:')
            res_a, res_b = res.split('Question: Which response is better? Response')[0].split('Response B:')
            print('===========')
            print(context.strip())
            print('A:', res_a.strip())
            print('B:', res_b.strip())
            print(row['PVI'], row['references_labels'], row['x_predicted_labels'])
            if row['cond_raw'] != ' ':
                print('Cond:', row['cond_raw'], row['cond_predicted_labels'])
        else:
            print(row['PVI'])
            print(row['x_raw'])
            print(row['references_labels'])
            print(row['x_predicted_labels'], row['cond_predicted_labels'])
            print('===========')
            pvis.append(row['PVI'])

    print(np.mean(pvis))

    # print the number of examples where cond_raw is 0.0
    # print('Number of examples where cond_raw is 0.0:', len(df[df['cond_raw'] == 0.0]))  # 272 for score
    # print(len(df))

if __name__ == '__main__':
    print_low_pvi_exs(harmless_vinfo)