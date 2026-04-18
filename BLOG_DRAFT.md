# A Pre-Trained Bag-of-Documents Model for Product Search

What if we represented search queries not by focusing on the query tokens, but in terms of the results those queries target? A query like "wireless keyboard" is not about the tokens "wireless" and "keyboard"; it maps to a distribution of relevant products. The centroid of that distribution captures the focus of the query intent, while the spread captures its specificity.

This is the essence of the [bag-of-documents model](https://dtunkelang.medium.com/modeling-queries-as-bags-of-documents-b7d79d0916ab). We train a model to predict these centroids from query text, and we obtain a retrieval encoder that understands product search.

[Aritra Mandal](https://www.linkedin.com/in/aritram/) and I developed this model together and shared details about our approach at a KDD 2023 workshop on e-commerce and NLP. Since then, I have been gratified to see search teams leverage this approach, in whole or in part.

## Democratizing the Bag-of-Documents Model

However, democratizing e-commerce search means making it work for organizations that lack the massive data to train a robust model. While no single approach can meet everyone's needs, a broad, deep pretrained model can give most organizations a solid head start.

To build such a model, we leverage resources that others before us had been kind to freely share. Namely:

- A collection of 30+ million products across all 33 Amazon categories from the [McAuley Lab Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/).
- 75K real search queries from the [Amazon Shopping Queries Dataset](https://arxiv.org/abs/2206.06588).
- A [cross-encoder regression model](https://huggingface.co/LiYuan/Amazon-Cup-Cross-Encoder-Regression) that Li Yuan developed for the Amazon Shopping Queries Dataset.
- The 384-dimensional [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) sentence transformers model as a base embedding model for product titles.
- The [FAISS](https://github.com/facebookresearch/faiss) similarity search index.
- The [tantivy](https://github.com/quickwit-oss/tantivy) full-text search engine library.

This free, open-source pipeline runs on a 16GB MacBook Air M4 laptop — though I did take a 20% sample of the product data so that the FAISS product index would fit into the available memory.

On Apple Silicon, the cross-encoder (a 6-layer RoBERTa model) benefits from MPS GPU acceleration — but only at large batch sizes. At batch size 8, CPU was faster (GPU data transfer overhead dominated). At batch size 32, MPS was 25% faster. We settled on batch size 32 with MPS, achieving ~64 bags/minute on a MacBook with 16GB RAM.

## Populating the Bags

The process for populating the bag of documents for a query is a simple hybrid retrieval process with a relevance filter:

1. Embed the query using the base MiniLM model and use FAISS to retrieve the products with vectors most similar to it.
2. Use tantivy to retrieve results with matching keywords.
3. Keep the results that score at least 0.3 in the cross-encoder, which is trained using the Amazon ESCI benchmark. The 0.3 threshold is somewhat arbitrary, so it is worth experimenting with other values.

Once we have the results for each bag, we use them to compute and store the centroid and specificity. We store the titles of the products in each bag but only keep the centroid vector and specificity, discarding the individual product vectors to avoid blowing up disk space.

We considered dropping keyword search entirely and using only FAISS embedding retrieval, letting the cross-encoder handle quality. However, we found that over a third of bag members came from keyword search only, and that just over half came from FAISS embedding search only. Indeed, there was minimal overlap. Keyword search does a good job of catching exact brand and model name matches, while FAISS excels at catching semantic matches without keyword overlap. Hybrid retrieval, guarded by the cross-encoder relevance filter, maximizes both precision and recall.

I experimented with a variety of filtering heuristics, but ultimately dropped all of them. Just using the cross-encoder was not only simpler and more principled, but also yielded the best results.

## Fine-Tuning

*[TODO: describe MSE loss training on bag centroids, base MiniLM, results]*

## Iterative Training

*[TODO: if iterative refinement works, describe re-embedding with fine-tuned model → recompute bags → fine-tune again]*

## Results

*[TODO: fill in broad catalog numbers]*

| Metric | Base MiniLM | Fine-tuned |
|--------|-------------|------------|
| Cosine sim to centroids | | |
| Recall@10 | | |
| ESCI precision | | |
| Complement retrieval rate | | |

## Resources

The fine-tuned model is available on HuggingFace at *[TODO]*, along with the bag of documents used to train it. The code is available at [github.com/dtunkelang/bag-of-documents](https://github.com/dtunkelang/bag-of-documents), and it not only allows you to reproduce the pipeline, but also to apply it using different parameters, models, and data.

## Next Steps

This release is exciting, but I hope it is only the first step in this journey. I am exploring scaling the pipeline to use the full 30M-product catalog — which is too large to fit on a laptop. I am also exploring whether a different base model or cross-encoder can improve results further. Finally, I hope that everyone who uses the model will contribute their learnings back to the community.

## Acknowledgments

This project is based on work by [Daniel Tunkelang](https://www.linkedin.com/in/dtunkelang/) and [Aritra Mandal](https://www.linkedin.com/in/aritram/).
