import pickle
import data

feature = pickle.load(open('../BERT/max_f1/coqa-train-v1.0.json_bert-base-uncased_512_128_385_2_coqa', 'rb'))

evidence_info = {}
for item in feature:
    evidence_info[item.qas_id] = {'doc_span': item.doc_span_index, 'sentence_id': item.sentence_id}

pickle.dump(evidence_info, open('../BERT/max_f1/evidence_id.pkl', 'wb'))

