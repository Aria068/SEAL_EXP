"""
Authorship Attribution Attack Evaluation

This script evaluates whether authorship can still be attributed after text anonymization.
It tests if stylometric fingerprints survive the SEAL rewriting process by:
1. Splitting each author's texts into train/query sets
2. Computing style embeddings for anonymized texts
3. Performing similarity-based retrieval
4. Measuring if queries match their correct authors

Usage:
    python eval/evaluate_authorship.py \
        --data_file data/main-train.jsonl \
        --min_texts_per_author 5 \
        --train_ratio 0.9 \
        --text_variant 0 \
        --embedding_model sentence-transformers/all-mpnet-base-v2 \
        --top_k 1,5,10

    python eval/evaluate_authorship.py \
        --data_file data/main-train.jsonl \
        --min_texts_per_author 5 \
        --train_ratio 0.9 \
        --text_variant 0 \
        --embedding_model AnnaWegmann/Style-Embedding \
        --top_k 1,5,10
    
    python eval/evaluate_authorship.py \
        --data_file data/main-train.jsonl \
        --min_texts_per_author 5 \
        --train_ratio 0.9 \
        --text_variant 0 \
        --embedding_model models/authorship-mixed-attack \
        --top_k 1,5,10
    
"""

import argparse
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class AuthorData:
    """Store author's train and query texts"""
    author_id: int
    train_texts: List[Tuple[str, str]]  # List of (text_id, text_content)
    query_texts: List[Tuple[str, str]]  # List of (text_id, text_content)


class AuthorshipEvaluator:
    """Evaluates authorship attribution attacks on anonymized texts"""

    def __init__(
        self,
        data_file: str,
        min_texts_per_author: int = 5,
        train_ratio: float = 0.7,
        text_variant: int = 0,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        top_k: List[int] = [1, 5, 10],
        seed: int = 42
    ):
        self.data_file = data_file
        self.min_texts_per_author = min_texts_per_author
        self.train_ratio = train_ratio
        self.text_variant = text_variant  # Which text variant to use (0=original, 1+=anonymized)
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.seed = seed
        np.random.seed(seed)

        self.authors_data: Dict[int, AuthorData] = {}
        self.model = None

    def load_and_filter_data(self):
        """Load JSONL data and filter authors with sufficient texts"""
        print(f"Loading data from {self.data_file}...")

        # Group texts by author
        author_texts = defaultdict(list)
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                author_id = record['author']
                text_id = record['id']

                # Extract the specified text variant
                if self.text_variant >= len(record['texts']):
                    print(f"Warning: text_variant {self.text_variant} not available for record {text_id}, using last variant")
                    text_content = record['texts'][-1]['text']
                else:
                    text_content = record['texts'][self.text_variant]['text']

                author_texts[author_id].append((text_id, text_content))

        # Filter authors with minimum number of texts
        print(f"Filtering authors with at least {self.min_texts_per_author} texts...")
        filtered_authors = {
            author_id: texts
            for author_id, texts in author_texts.items()
            if len(texts) >= self.min_texts_per_author
        }

        print(f"Total authors before filtering: {len(author_texts)}")
        print(f"Authors after filtering: {len(filtered_authors)}")
        print(f"Average texts per author: {np.mean([len(v) for v in filtered_authors.values()]):.1f}")

        return filtered_authors

    def split_train_query(self, author_texts: Dict[int, List[Tuple[str, str]]]):
        """Split each author's texts into train and query sets"""
        print(f"Splitting texts into train ({self.train_ratio:.0%}) and query sets...")

        for author_id, texts in author_texts.items():
            # Shuffle texts for this author
            shuffled_texts = texts.copy()
            np.random.shuffle(shuffled_texts)

            # Split into train and query
            n_train = max(1, int(len(texts) * self.train_ratio))
            train_texts = shuffled_texts[:n_train]
            query_texts = shuffled_texts[n_train:]

            # Only include authors who have both train and query texts
            if len(query_texts) > 0:
                self.authors_data[author_id] = AuthorData(
                    author_id=author_id,
                    train_texts=train_texts,
                    query_texts=query_texts
                )

        total_train = sum(len(ad.train_texts) for ad in self.authors_data.values())
        total_query = sum(len(ad.query_texts) for ad in self.authors_data.values())
        print(f"Final authors with both train and query: {len(self.authors_data)}")
        print(f"Total train texts: {total_train}")
        print(f"Total query texts: {total_query}")

    def load_embedding_model(self):
        """Load the sentence embedding model"""
        print(f"Loading embedding model: {self.embedding_model_name}...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.embedding_model_name)
            print("Model loaded successfully")
        except ImportError:
            print("ERROR: sentence-transformers not installed")
            print("Please run: pip install sentence-transformers")
            raise

    def compute_embeddings(self):
        """Compute embeddings for all train and query texts"""
        print("Computing embeddings for all texts...")

        # Collect all train texts
        train_texts = []
        train_metadata = []  # (author_id, text_id)
        for author_id, author_data in self.authors_data.items():
            for text_id, text_content in author_data.train_texts:
                train_texts.append(text_content)
                train_metadata.append((author_id, text_id))

        # Collect all query texts
        query_texts = []
        query_metadata = []  # (author_id, text_id)
        for author_id, author_data in self.authors_data.items():
            for text_id, text_content in author_data.query_texts:
                query_texts.append(text_content)
                query_metadata.append((author_id, text_id))

        # Compute embeddings in batches
        print(f"Computing {len(train_texts)} train embeddings...")
        train_embeddings = self.model.encode(
            train_texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        print(f"Computing {len(query_texts)} query embeddings...")
        query_embeddings = self.model.encode(
            query_texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        return train_embeddings, train_metadata, query_embeddings, query_metadata

    def evaluate_retrieval(
        self,
        train_embeddings: np.ndarray,
        train_metadata: List[Tuple[int, str]],
        query_embeddings: np.ndarray,
        query_metadata: List[Tuple[int, str]]
    ) -> Dict:
        """Perform retrieval and compute metrics"""
        print("Performing similarity-based retrieval...")

        # Compute cosine similarity matrix (queries x train)
        # Normalize embeddings for cosine similarity
        train_embeddings_norm = train_embeddings #/ np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        query_embeddings_norm = query_embeddings #/ np.linalg.norm(query_embeddings, axis=1, keepdims=True)

        # Similarity matrix: (n_queries, n_train)
        similarity_matrix = query_embeddings_norm @ train_embeddings_norm.T

        # For each query, get top-k most similar train texts
        results = []
        for query_idx, (query_author, query_text_id) in enumerate(tqdm(query_metadata, desc="Evaluating queries")):
            # Get similarity scores for this query
            scores = similarity_matrix[query_idx]

            # Sort by similarity (descending)
            sorted_indices = np.argsort(-scores)

            # Get top-k results
            top_k_results = []
            for rank, train_idx in enumerate(sorted_indices[:max(self.top_k)], start=1):
                train_author, train_text_id = train_metadata[train_idx]
                similarity = float(scores[train_idx])
                top_k_results.append({
                    'rank': rank,
                    'train_author': train_author,
                    'train_text_id': train_text_id,
                    'similarity': similarity,
                    'is_correct': train_author == query_author
                })

            results.append({
                'query_author': query_author,
                'query_text_id': query_text_id,
                'retrievals': top_k_results
            })

        return self.compute_metrics(results)

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute evaluation metrics from retrieval results"""
        print("\nComputing metrics...")

        metrics = {
            'accuracy': {},  # Accuracy@k
            'recall': {},    # Recall@k
            'mrr': 0.0,      # Mean Reciprocal Rank
            'total_queries': len(results)
        }

        reciprocal_ranks = []

        for result in results:
            query_author = result['query_author']
            retrievals = result['retrievals']

            # Find first correct match
            first_correct_rank = None
            for retrieval in retrievals:
                if retrieval['is_correct']:
                    first_correct_rank = retrieval['rank']
                    break

            # MRR
            if first_correct_rank:
                reciprocal_ranks.append(1.0 / first_correct_rank)
            else:
                reciprocal_ranks.append(0.0)

            # Accuracy@k and Recall@k
            for k in self.top_k:
                top_k_retrievals = [r for r in retrievals if r['rank'] <= k]

                # Accuracy@k: Is there at least one correct match in top-k?
                has_correct = any(r['is_correct'] for r in top_k_retrievals)
                if k not in metrics['accuracy']:
                    metrics['accuracy'][k] = []
                metrics['accuracy'][k].append(1.0 if has_correct else 0.0)

                # Recall@k: What fraction of correct matches are in top-k?
                # (For authorship, there can be multiple texts from same author in train set)
                if k not in metrics['recall']:
                    metrics['recall'][k] = []
                correct_in_topk = sum(1 for r in top_k_retrievals if r['is_correct'])
                # We need to know total possible correct matches for this query
                # This is the number of train texts from the same author
                total_correct = sum(1 for _, train_author, _ in
                                  [(r['rank'], r['train_author'], r['train_text_id']) for r in retrievals]
                                  if train_author == query_author)
                recall = correct_in_topk / total_correct if total_correct > 0 else 0.0
                metrics['recall'][k].append(recall)

        # Average metrics
        metrics['mrr'] = np.mean(reciprocal_ranks)
        for k in self.top_k:
            metrics['accuracy'][k] = np.mean(metrics['accuracy'][k])
            metrics['recall'][k] = np.mean(metrics['recall'][k])

        return metrics, results

    def print_metrics(self, metrics: Dict):
        """Print evaluation metrics"""
        print("\n" + "="*60)
        print("AUTHORSHIP ATTRIBUTION ATTACK EVALUATION RESULTS")
        print("="*60)
        print(f"Dataset: {self.data_file}")
        print(f"Text variant: {self.text_variant} ({'original' if self.text_variant == 0 else f'anonymized-{self.text_variant}'})")
        print(f"Embedding model: {self.embedding_model_name}")
        print(f"Authors: {len(self.authors_data)}")
        print(f"Total queries: {metrics['total_queries']}")
        print("\nMetrics:")
        print(f"  MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
        for k in self.top_k:
            print(f"  Accuracy@{k}: {metrics['accuracy'][k]:.4f} ({metrics['accuracy'][k]*100:.2f}%)")
            print(f"  Recall@{k}: {metrics['recall'][k]:.4f} ({metrics['recall'][k]*100:.2f}%)")
        print("="*60)

    def save_results(self, metrics: Dict, results: List[Dict], output_file: str):
        """Save detailed results to JSON file"""
        output_data = {
            'config': {
                'data_file': self.data_file,
                'min_texts_per_author': self.min_texts_per_author,
                'train_ratio': self.train_ratio,
                'text_variant': self.text_variant,
                'embedding_model': self.embedding_model_name,
                'top_k': self.top_k,
                'seed': self.seed
            },
            'metrics': {
                'mrr': metrics['mrr'],
                'accuracy': {str(k): v for k, v in metrics['accuracy'].items()},
                'recall': {str(k): v for k, v in metrics['recall'].items()},
                'total_queries': metrics['total_queries'],
                'num_authors': len(self.authors_data)
            },
            'detailed_results': results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {output_file}")

    def run(self, output_file: str = None):
        """Run the full evaluation pipeline"""
        # 1. Load and filter data
        author_texts = self.load_and_filter_data()

        # 2. Split train/query
        self.split_train_query(author_texts)

        if len(self.authors_data) == 0:
            print("ERROR: No authors with sufficient texts for train/query split")
            return

        # 3. Load embedding model
        self.load_embedding_model()

        # 4. Compute embeddings
        train_embeddings, train_metadata, query_embeddings, query_metadata = self.compute_embeddings()

        # 5. Evaluate retrieval
        metrics, results = self.evaluate_retrieval(
            train_embeddings, train_metadata,
            query_embeddings, query_metadata
        )

        # 6. Print and save results
        self.print_metrics(metrics)

        if output_file:
            self.save_results(metrics, results, output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate authorship attribution attacks on anonymized texts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/main-train.jsonl",
        help="Path to JSONL data file"
    )
    parser.add_argument(
        "--min_texts_per_author",
        type=int,
        default=5,
        help="Minimum number of texts required per author"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of texts to use for training (rest for query)"
    )
    parser.add_argument(
        "--text_variant",
        type=int,
        default=0,
        help="Which text variant to use (0=original, 1+=anonymized versions)"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence transformer model to use for embeddings"
    )
    parser.add_argument(
        "--top_k",
        type=str,
        default="1,5,10",
        help="Comma-separated list of k values for top-k metrics"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save detailed results (auto-generated if not specified)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Parse top_k
    top_k = [int(k.strip()) for k in args.top_k.split(',')]

    # Auto-generate output filename if not specified
    if args.output_file is None:
        import os
        base_name = os.path.basename(args.data_file).replace('.jsonl', '')
        variant_name = 'original' if args.text_variant == 0 else f'anon{args.text_variant}'
        model_name = args.embedding_model.split('/')[-1]
        args.output_file = f"eval/authorship_{base_name}_{variant_name}_{model_name}.json"

    # Run evaluation
    evaluator = AuthorshipEvaluator(
        data_file=args.data_file,
        min_texts_per_author=args.min_texts_per_author,
        train_ratio=args.train_ratio,
        text_variant=args.text_variant,
        embedding_model=args.embedding_model,
        top_k=top_k,
        seed=args.seed
    )

    evaluator.run(output_file=args.output_file)


if __name__ == "__main__":
    main()
