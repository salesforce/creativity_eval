import numpy as np
from scipy import stats
import json

def calculate_kendalls_w(data):
    """
    Calculate Kendall's W for preference rankings.
    
    :param data: List of lists, where each sublist contains preference rankings from all annotators for one instance.
    :return: Kendall's W value
    """
    if not data or not all(data):
        raise ValueError("Input data is empty or contains empty lists")

    n_instances = len(data)
    n_annotators = len(data[0])
    n_items = len(data[0][0])
    
    # Validate input data
    if not all(len(instance) == n_annotators for instance in data):
        raise ValueError("Inconsistent number of annotators across instances")
    if not all(len(annotator_ranking) == n_items for instance in data for annotator_ranking in instance):
        raise ValueError("Inconsistent number of items in rankings")

    # Convert rankings to ranks
    ranked_data = []
    for instance in data:
        instance_ranks = []
        for annotator_ranking in instance:
            # Create a mapping of item to its position (rank)
            rank_map = {item: rank for rank, item in enumerate(sorted(set(annotator_ranking)), 1)}
            # Convert each item to its rank
            ranks = [rank_map[item] for item in annotator_ranking]
            instance_ranks.append(ranks)
        ranked_data.append(instance_ranks)

    w_values = []
    
    for instance in ranked_data:
        rankings = np.array(instance)
        correlations = []
        for i in range(n_annotators):
            for j in range(i+1, n_annotators):
                tau, _ = stats.kendalltau(rankings[i], rankings[j])
                correlations.append(tau)
        w_values.append(np.mean(correlations))
    
    return np.mean(w_values)


#label_map = {'Human-edited': 1, 'AI-generated': 2, 'AI-edited': 3}

with open('agreement_map.json') as f:
    data = json.load(f)


for batch in data:
    workers = []
    for worker in data[batch]:
        workers.append(worker)
    d = []
    for a, b, c in zip(data[batch][workers[0]],data[batch][workers[1]],data[batch][workers[2]]):
        d.append([a,b,c])
    try:
        w = calculate_kendalls_w(d)
    except ValueError as e:
        print(f"Error: {e}")
    print(batch,w)

