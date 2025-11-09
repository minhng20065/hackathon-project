from transformers import BertTokenizer
from dataclean import CEAS_08
from tqdm import tqdm
import time
from phishing_detector import NonMLPhishingDetector
URGENCY_KEYWORDS = [
    'urgent', 'asap', 'immediately', 'emergency', 'critical', 'right away',
        'now', 'quick', 'fast', 'rush', 'deadline', 'time sensitive',
        'important', 'priority', 'urgently', 'stat', '911', 'emergency',
        'breaking', 'crisis', 'alert', 'attention needed', 'soon', 'shortly', 'prompt', 'timely', 'expedite', 'hurry',
        'quickly'
]
new_df = CEAS_08.head(10)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
URGENCY_KEYWORDS_SET = set(URGENCY_KEYWORDS)
print(f"Processing {len(new_df)} items....")
start_time = time.time()

# Use tqdm for progress bar
urgency_scores = []

for i in tqdm(range(len(new_df)), desc="Processing urgency"):
    text = str(new_df.iloc[i, 1])
    tokens = tokenizer.tokenize(text)
    urgency = sum(1 for keyword in URGENCY_KEYWORDS_SET if keyword in tokens)
    detector = NonMLPhishingDetector()
    urgency_scores.append(urgency)

new_df['urgency'] = urgency_scores

end_time = time.time()
newCS = new_df
print(newCS.columns)
