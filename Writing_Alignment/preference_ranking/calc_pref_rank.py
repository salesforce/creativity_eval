
import json
from scipy.stats import wilcoxon

oracle = []
pred = []

with open('./LLM_edited_full.json') as f:
    data = json.load(f)

pred = [elem['id'] for elem in data]

with open('./LLM_edited_oracle.json') as f:
    data = json.load(f)

oracle = [elem['id'] for elem in data]

print(set(oracle).intersection(set(pred)))

mp = {
    'batch1': ['cgreer', 'yen', 'issa'],
    'batch2': ['riley', 'xiadi', 'rachel_lapides'],
    'batch3': ['hyo', 'margaret', 'imani'],
    'batch4': ['cgreer', 'josiah', 'yen'],
    'batch5': ['adeniyiademoroti', 'cgreer', 'rachel_lapides'],
    'batch6': ['cgreer', 'josiah', 'yen'],
    'batch7': ['stefan', 'micah', 'hyo']
}
ranks = {
    1: {'AI-generated': 0, 'Human-edited': 0, 'AI-edited': 0},
    2: {'AI-generated': 0, 'Human-edited': 0, 'AI-edited': 0},
    3: {'AI-generated': 0, 'Human-edited': 0, 'AI-edited': 0}
}

overall = 0
ai_generated_ranks = []
ai_edited_ranks = []
human_edited_ranks = []



for batch in mp:
    num = 0
    for worker in mp[batch]:
        with open('./' + batch + '/' + worker + '.json') as f:
            data = json.load(f)

        x = {elem['instruction_id']: elem['mapping'] for elem in data}
        ids = {elem['instruction_id']: elem['id'] for elem in data}

        with open('./' + batch + '/' + worker + '_rankings.json') as f:
            ranking = json.load(f)

        for elem in ranking:
            iden = ids[elem['instruction_id']]
            
            #Calculating preference ranking on LLM edited oracle. To change it to LLM edited full replace oracle with pred
            if iden not in oracle:
                continue

            overall += 1
            rank1 = elem['rank_1']
            rank2 = elem['rank_2']
            rank3 = elem['rank_3']

            # Collect rankings for AI-edited and AI-generated for statistical testing
            # ai_generated_ranks.append(1 if x[elem['instruction_id']][rank1] == 'AI-generated' else 2 if x[elem['instruction_id']][rank2] == 'AI-generated' else 3)
            ai_edited_ranks.append(1 if x[elem['instruction_id']][rank1] == 'AI-edited' else 2 if x[elem['instruction_id']][rank2] == 'AI-edited' else 3)
            ai_generated_ranks.append(1 if x[elem['instruction_id']][rank1] == 'AI-generated' else 2 if x[elem['instruction_id']][rank2] == 'AI-generated' else 3)
            human_edited_ranks.append(1 if x[elem['instruction_id']][rank1] == 'Human-edited' else 2 if x[elem['instruction_id']][rank2] == 'Human-edited' else 3)

            ranks[1][x[elem['instruction_id']][rank1]] += 1
            ranks[2][x[elem['instruction_id']][rank2]] += 1
            ranks[3][x[elem['instruction_id']][rank3]] += 1

# Perform Wilcoxon signed-rank test
stat, p_value = wilcoxon(ai_generated_ranks, human_edited_ranks)
print(f'Wilcoxon test statistic: {stat}, p-value: {p_value}')

stat, p_value = wilcoxon(ai_generated_ranks, ai_edited_ranks)
print(f'Wilcoxon test statistic: {stat}, p-value: {p_value}')

for k in ranks:
    print('Rank ' + str(k) + ' : ', ranks[k])

final_rank = {'AI-generated': 0, 'Human-edited': 0, 'AI-edited': 0}
for position in ranks:
    for k in ranks[position]:
        final_rank[k] = final_rank[k] + position * ranks[position][k]

for k in final_rank:
    final_rank[k] = round(float(final_rank[k]) / float(overall), 2)

print(final_rank)
print(overall)
