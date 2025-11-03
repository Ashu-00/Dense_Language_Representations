import json
import matplotlib.pyplot as plt

with open('evaluation_results.json', 'r') as f:
    data = json.load(f)
    metrics = data.keys() # simlex-999 and analogy accuracy

    simlex = data['SimLex-999']
    analogy = data['Analogy Task']
    clean_res = {}

    for model, value in simlex.items():
        if model.startswith('emb_'):
            dim = int(model.split('_')[1])
            window_size = int(model.split('_')[2])
            if window_size not in clean_res:
                clean_res[window_size] = {'dims': [], 'simlex': [], 'analogy': []}
            clean_res[window_size]['dims'].append(dim)
            clean_res[window_size]['simlex'].append(value)
            clean_res[window_size]['analogy'].append(analogy[model])
        else:
            if not 'pretrained' in clean_res:
                clean_res['pretrained'] = {'simlex': {}, 'analogy': {}}
            clean_res['pretrained']['simlex'][model] = value
            clean_res['pretrained']['analogy'][model] = analogy[model]


    plt.figure(figsize=(15, 12))
    plt.subplot(2,1,1)
    plt.title('SimLex-999 Correlation vs Embedding Dimension')
    for window_size, results in clean_res.items():
        if window_size == 'pretrained':
            continue
        plt.plot(results['dims'], results['simlex'], marker='o', label=f'Window Size {window_size}')
    # for model, value in clean_res['pretrained']['simlex'].items():
    #     plt.hlines(value, xmin=0, xmax=1000, linestyles='dashed', label=f'Pretrained: {model}')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Spearman Correlation')
    plt.grid()

    plt.legend()

    plt.subplot(2,1,2)
    plt.title('Analogy Task Accuracy vs Embedding Dimension')
    for window_size, results in clean_res.items():
        if window_size == 'pretrained':
            continue
        plt.plot(results['dims'], results['analogy'], marker='o', label=f'Window Size {window_size}')
    # for model, value in clean_res['pretrained']['analogy'].items():
    #     plt.hlines(value, xmin=0, xmax=1000, linestyles='dashed', label=f'Pretrained: {model}')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Analogy Accuracy')
    plt.legend()

    plt.grid()

    plt.savefig("evaluation_results.png")
            
