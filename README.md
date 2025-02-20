# Automated-File-Explorer

An intelligent document exploration system that uses LLMs (Language Model Models) to understand and answer questions about your documents.

## Features

- Supports multiple document formats (PDF, TXT, DOCX)
- Uses advanced language models for accurate responses
- Local execution for data privacy
- Vector-based document search
- Source attribution for answers

## Requirements

- Python 3.8+
- At least 8GB RAM
- 5GB free disk space
- Required Python packages:
  ```bash
  pip install langchain langchain-community langchain-text-splitters faiss-cpu transformers torch sentence-transformers python-docx pypdf
  ```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Automated-File-Explorer.git
   cd Automated-File-Explorer
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv filemanager
   # On Windows
   filemanager\Scripts\activate
   # On Unix or MacOS
   source filemanager/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `docs` folder in the project root and add your documents:
   ```bash
   mkdir docs
   # Add your PDF, TXT, or DOCX files to the docs folder
   ```

## Usage

1. Place your documents in the `docs` folder
2. Run the application:
   ```bash
   python app.py
   ```

3. The system will:
   - Load and process your documents
   - Create a searchable index
   - Answer questions about your documents

## First Run

On first run, the system will:
1. Download required models (~4GB)
2. Create a vector store index of your documents
3. This may take 10-15 minutes depending on your internet connection and system specs

## Configuration

You can modify these parameters in `app.py`:

- `chunk_size`: Size of document chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `model_name`: Language model to use (default: "microsoft/phi-3-mini-4k-instruct")

## Supported File Types

- PDF (`.pdf`)
- Text files (`.txt`)
- Word documents (`.docx`)

## Performance Notes

- First run will download models and create indexes
- Subsequent runs will use cached models
- Query processing time depends on:
  - Document size and number
  - System resources
  - Model size

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed
2. Check if `docs` directory exists and contains documents
3. Verify sufficient disk space and RAM
4. For slow queries:
   - Reduce number of documents
   - Use a smaller model
   - Reduce chunk size

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request