"""Prepare and verify the pinned public checkpoint and original tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from huggingface_hub import snapshot_download
import sentencepiece
from transformers import AutoTokenizer, GemmaTokenizerFast


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=Path('/cache'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    configuration = json.loads((ROOT / 'config/model_assets.json').read_text())
    checkpoint = args.cache / 'model'
    tokenizer_directory = args.cache / 'tokenizer'
    vocabulary = tokenizer_directory / 'tokenizer.model'
    if not args.verify_only:
        print('Model terms: licenses/Gemma-Terms.txt and Gemma-Prohibited-Use-Policy.txt')
        snapshot_download(**configuration['model'], local_dir=checkpoint, token=False,
                          max_workers=2, allow_patterns=[
                              'config.json', 'train_config.json', 'model.safetensors',
                              'README.md', 'LICENSE', 'NOTICE'])
        tokenizer_directory.mkdir(parents=True, exist_ok=True)
        if not vocabulary.exists():
            temporary = vocabulary.with_suffix('.partial')
            with urllib.request.urlopen(configuration['tokenizer']['url'], timeout=60) as response:
                temporary.write_bytes(response.read())
            if sha256(temporary) != configuration['tokenizer']['sha256']:
                raise RuntimeError('Tokenizer download failed SHA-256 verification.')
            temporary.replace(vocabulary)
    if sha256(checkpoint / 'model.safetensors') != configuration['model_sha256']:
        raise RuntimeError('Checkpoint SHA-256 does not match the pinned model.')
    if sha256(vocabulary) != configuration['tokenizer']['sha256']:
        raise RuntimeError('Original tokenizer SHA-256 does not match.')
    if not args.verify_only:
        tokenizer = GemmaTokenizerFast(vocab_file=str(vocabulary), add_bos_token=True,
                                       add_eos_token=False, padding_side='right')
        tokenizer.save_pretrained(tokenizer_directory)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory, local_files_only=True)
    reference = sentencepiece.SentencePieceProcessor(model_file=str(vocabulary))
    prompt = 'Task: ' + configuration['instruction'] + ';\nAction: '
    if tokenizer.encode(prompt) != [reference.bos_id()] + reference.encode(prompt):
        raise RuntimeError('Tokenizer IDs differ from the original SentencePiece vocabulary.')
    print('Pinned model and original tokenizer verified.')


if __name__ == '__main__':
    main()
