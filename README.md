2. We would also like to ask you to implement a method to identify the language a document is written in. You can use any framework available, but you should implement an approach on your own. If you re-implement an approach taken from the literature, please provide a reference. The suggested programming language is Python.

Input:
- Plain text input
- .txt
- .pdf: text and OCR
- .doc/docx

Output:
- list of possible languages, with probability

App:
- Input:
    - File upload
    - Text input
    - Batch upload
- Model selection
    - Simple model
    - Transformer-based model
- Output
    - A table

- Deployment
    - Docker
    - HuggingFace deployment?


https://huggingface.co/FacebookAI/xlm-roberta-base


Backend:
- Input processor:
    - Receive different file types & raw text
        - raw text processor
        - txt processor
        - doc/docx processor
        - pdf processor:
            - Handle pdf with text layer
            - Handle scanned pdf
        - image processor
    - Features:
        - read/parse text, tokenize, embedding
        - output a universal type
- Predictor
    - Pipelines:
        - raw text - sklearn pipeline
        - raw text - transformer pipeline
    - Input handler:
        - sklearn pipeline: sklearn-compatible matrix
        - transformer pipeline: transformer-compatible matrix