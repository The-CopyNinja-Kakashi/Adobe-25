 
  
# Adobe Hackathon Task 1A: PDF Outline Extraction

A high-performance and highly accurate CPU-optimized solution that uses a hybrid machine learning and rule-based approach to extract structured outlines (topics and headings) from PDF documents.

## Overview

This solution is a high-performance, CPU-optimized system designed to extract structured outlines — including topics and headings — from PDF documents. It leverages a hybrid machine learning and rule-based approach:

A Random Forest classifier analyzes layout features (e.g - font size, boldness, indentation, spacing) to accurately identify topics and headings.

If the ML model produces no results, a rule-based fallback parser automatically extracts the largest or boldest text as the topic and builds headings from the remaining structure.

This dual-layer strategy ensures robust, accurate outline extraction across diverse PDF formats (academic papers, textbooks, business reports) while staying within strict CPU-only and lightweight runtime constraints.

## Features

- **Hybrid ML + Rule-Based Parsing**: Uses a trained Random Forest model for high-accuracy topic & heading classification, with a fallback parser for PDFs where layout is inconsistent.
- **Fast Performance**: Optimized for ≤10 second processing of 50-page PDFs
- **CPU Optimized**: Runs efficiently on AMD64 architecture without GPU requirements
- **Double-Level Design**: Provides two layers of intelligence:
Machine Learning layer (Random Forest classifier for precise topic & heading detection)
Fallback layer (rule-based parser for resilience on messy layouts).
- **Fail-Safe Parsing**: If the ML model returns no results, the fallback method ensures   that at least a meaningful outline is extracted. 
- **Docker Ready**: Containerized solution with all dependencies included

## Architecture

### Method 1:  ML-Driven Heading Detection (Primary)
- Uses a trained Random Forest classifier on layout features (font size, boldness, spacing, indentation, etc.)
- Accurately distinguishes between topics, headings, and regular text
- Works across diverse PDF types – research papers, textbooks, reports

### Method 2: Rule-Based Fallback Parsing
- AWhen the ML model returns low/no confidence results, switches to a font-based heuristic parser
- Detects potential headings by comparing font hierarchy, boldness, and relative positioning
- Guarantees extraction even from messy PDFs with inconsistent formatting

### Method 3: Title & Contextual Outline Extraction
- Derives document titles either from detected “topic” class or largest bold text
- Enriches the extracted headings with page numbers and hierarchical structure
- Ensures even simple PDFs (like letters or memos) get a meaningful outline

## Algorithm Details

### ML + Feature-Based Classification
1. **Feature Engineering**: Extracts 20+ layout features (font size, bold flags, indentation, spacing, capitalization ratio, etc.).
2. **Random Forest Model**: Trained to classify lines into topic, heading, or body classes.
3. **Confidence Thresholding**: If confidence is low or model returns sparse results, pipeline switches to fallback mode
4. **Position-Based Filtering**: Considers text positioning and formatting

### Font Analysis Heuristics (Fallback Mode)
- **Statistical Font Analysis**: Identifies the most common font size as “body text.”
- **Heading Font Detection**: Finds text lines significantly larger or bolder than body text.
- **Position-Based Filtering**: Uses x/y positioning to reject page numbers, footers, or watermarks.

### Text Cleaning
- Removes page numbers, chapter prefixes (e.g., “1.1”, “CHAPTER 3”) and formatting artifacts
- VValidates heading text length (keeps only lines between 3–200 characters).
- Normalizes whitespace, merges broken headings, and strips special characters for cleaner output.

## Dependencies

- **PyMuPDF**: Fast PDF processing and outline extraction
- **scikit-learn**: Machine learning framework for the Random Forest classifier that identifies topics and headings.
- **pandas**: Handles structured tabular data for feature extraction and ML input.
- **numpy**: Efficient numerical operations for font size statistics, spacing calculations, and feature scaling.


## Docker Usage

### Build the Image
### COMMAND TO BUILD IMAGE ON POWERSHELL

docker build `
  --platform linux/amd64 `
  -t pdf-parser `
  .



### Run the Container
### POWERSHELL RUN COMMAND
 docker run --rm `
   -v "${PWD}\sample_dataset\pdfs:/app/sample_dataset/pdfs" `
   -v "${PWD}\sample_dataset\output:/app/sample_dataset/output" `
   -v "${PWD}\finalmodel.pkl:/app/finalmodel.pkl" `
   --network none `
   pdf-parser


## Input/Output Format

### Input
- Directory: `/app/sample_dataset/pdfs`
- Format: PDF files (*.pdf)
- Limit: Up to 50 pages per PDF

### Output
- Directory: `/app/sample_dataset/output`
- Format: JSON files with same name as input PDF
- Structure:
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction", 
      "page": 1
    },
    {
      "level": "H2",
      "text": "Background",
      "page": 2
    }
  ]
}
```

## Performance Specifications

- **Processing Time**: ≤10 seconds for 50-page PDF
- **Model Size**: Lightweight python libraries. Small ensemble model used(<10 mb).
- **Memory Usage**: Optimized for 16GB RAM systems
- **CPU**: Utilizes multi-core processing on 8-CPU systems
- **Architecture**: AMD64 (x86_64) compatible

## Error Handling

- Graceful fallback between extraction methods
- Robust error logging and recovery
- Empty results for failed extractions rather than crashes

## Optimization Features

- **Lazy Loading**: Processes pages only when needed
- **Memory Efficient**: Closes resources promptly
- **Parallel Ready**: Can be extended for multi-file parallel processing
- **Caching**: Reuses font analysis across pages

## Testing

The solution has been designed to handle various PDF types:
- Academic papers with clear heading hierarchies
- Business documents with embedded bookmarks
- Scanned documents (where text is selectable)
- Multi-language documents
- Complex layouts with mixed formatting

## Limitations

- Requires selectable text (not pure image-based PDFs)
- Heading detection accuracy depends on consistent font usage
- Limited to H1, H2, H3 levels as per requirements
- No network access for enhanced processing

## Future Enhancements

- Advanced layout analysis with computer vision.
- Deep Learning Based enhanced classification.
- Support for tables and figures in outline.
- Improved accuracy using robust dataset and models.