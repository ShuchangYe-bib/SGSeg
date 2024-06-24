import json
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import HDBSCAN, KMeans


ann_path = "annotation.json"

locations = []

annotations = json.loads(open(ann_path, 'r').read())
for mode in annotations:
    for anno in annotations[mode]:
        caption = anno["caption"]
        location = caption.split(", ")[-1].strip()
        locations.append(location)


model_name = "microsoft/BiomedVLP-CXR-BERT-specialized"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name,output_hidden_states=True,trust_remote_code=True)

def get_bert_embeddings(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {key: val.to(model.device) for key, val in encoded_input.items()}
    with torch.no_grad():
        outputs = model(**encoded_input)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

embeddings = get_bert_embeddings(locations)
assert len(embeddings) == len(locations)
length = len(embeddings)

# Clustering with HDBSCAN
cluster = HDBSCAN(min_cluster_size=3)
# cluster = KMeans(n_clusters=64)
pseudo_labels = list(cluster.fit_predict(embeddings))

label_to_report = {}
report_to_label = {}

for i in range(length):
    label = int(pseudo_labels[i])
    report = locations[i]
    if label in label_to_report:
        label_to_report[label] += [report]
        label_to_report[label] = list(set(label_to_report[label]))
    else:
        label_to_report[label] = [report]
    if report in report_to_label:
        report_to_label[report] += [label]
        report_to_label[report] = list(set(report_to_label[report]))
    else:
        report_to_label[report] = [label]


print("the number of pseudo_labels: ", len(set(pseudo_labels)))
print(list(set(pseudo_labels)))

with open("label2report.json", "w") as l2r: 
    json.dump(label_to_report, l2r, indent=2)

with open("report2label.json", "w") as r2l: 
    json.dump(report_to_label, r2l, indent=2)

# csv_writer.writerow(header + ["Stage1", "Stage2", "Stage3"] + ["Pseudo"])
# for data_no in range(len(data)):
#     record = data[data_no]
#     csv_writer.writerow(record + record[1].lower().split(",") + [pseudo_labels[data_no]])

# output_csv.close()



