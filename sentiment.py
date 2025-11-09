from transformers import BertTokenizer
from dataclean import CEAS_08
from tqdm import tqdm
import time
URGENCY_KEYWORDS = [
    'urgent', 'asap', 'immediately', 'emergency', 'critical', 'right away',
        'now', 'quick', 'fast', 'rush', 'deadline', 'time sensitive',
        'important', 'priority', 'urgently', 'stat', '911', 'emergency',
        'breaking', 'crisis', 'alert', 'attention needed', 'soon', 'shortly', 'prompt', 'timely', 'expedite', 'hurry',
        'quickly'
]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
URGENCY_KEYWORDS_SET = set(URGENCY_KEYWORDS)
print(f"Processing {len(CEAS_08)} items....")
start_time = time.time()

# Use tqdm for progress bar
urgency_scores = []

for i in tqdm(range(len(CEAS_08)), desc="Processing urgency"):
    text = str(CEAS_08.iloc[i, 1])
    tokens = tokenizer.tokenize(text)
    urgency = sum(1 for keyword in URGENCY_KEYWORDS_SET if keyword in tokens)
    urgency_scores.append(urgency)

CEAS_08['urgency'] = urgency_scores

end_time = time.time()

print(CEAS_08.iloc[:5])
