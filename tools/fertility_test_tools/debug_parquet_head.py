import pyarrow.parquet as pq
import sys
import unicodedata
import pprint

parquet_path = './datasets/oscar-corpus_mOSCAR_swe_Latn.parquet'

def extract_text_from_record(record):
    text_fields = ['text', 'content', 'code', 'body', 'message']
    for field in text_fields:
        if field in record and isinstance(record[field], str):
            print(f"[EXTRACT] Field '{field}' found: {record[field][:80]!r}")
            return unicodedata.normalize("NFC", record[field])
        if field in record and isinstance(record[field], list):
            items = record[field]
            if all(isinstance(x, dict) and 'text' in x for x in items):
                joined = " ".join([str(x['text']) for x in items if isinstance(x, dict) and 'text' in x and x['text']])
                print(f"[EXTRACT] Field '{field}' is list of dicts, joined: {joined[:80]!r}")
                return unicodedata.normalize("NFC", joined)
            joined = " ".join([str(x) for x in items if isinstance(x, str)])
            print(f"[EXTRACT] Field '{field}' is list, joined: {joined[:80]!r}")
            return unicodedata.normalize("NFC", joined)
    for key, value in record.items():
        if isinstance(value, str):
            print(f"[EXTRACT] Any string field '{key}': {value[:80]!r}")
            return unicodedata.normalize("NFC", value)
    print(f"[EXTRACT] No text found in record: {record}")
    return None

try:
    table = pq.read_table(parquet_path, memory_map=True)
    count = 0
    for batch in table.to_batches(max_chunksize=10):
        for record in batch.to_pylist():
            print(f"\nRecord {count+1}:")
            text = extract_text_from_record(record)
            print(f"Extracted text: {repr(text)[:120]}")
            count += 1
            if count >= 5:
                sys.exit(0)
except Exception as e:
    print(f"Error reading Parquet file: {e}")
