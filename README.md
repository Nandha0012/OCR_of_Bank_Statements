# OCR Data Extraction and Visualization

## Overview
This project is a robust pipeline for extracting, processing, and visualizing data from scanned documents such as salary slips, balance slips, and cash slips. The system leverages OCR technology, a large language model (LLM), and MongoDB for data storage and retrieval. The application is user-friendly, featuring a Gradio interface for interaction.

## Features
- **OCR Processing**: Extracts text from images and PDFs using EasyOCR.
- **LLM Integration**: Utilizes ChatGroq with the LLaMA-3.1-70b-versatile model for text analysis and structured data extraction.
- **MongoDB Support**: Retrieves images stored in a MongoDB GridFS database.
- **Data Visualization**: Generates field-wise comparison charts (Pie and Bar charts) for extracted data.
- **Gradio Interface**: Provides an intuitive UI for uploading files, selecting document types, and viewing results.

## Installation

### Prerequisites
1. Python 3.8 or higher
2. MongoDB Atlas account (or local MongoDB setup)
3. Required Python libraries (see `requirements.txt`)
4. Gradio for UI

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ocr-data-visualization.git
   cd ocr-data-visualization
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up MongoDB:
   - Configure your MongoDB URI in the script.
   - Ensure the GridFS module is enabled.

4. Configure Gradio cache directory:
   - Modify the `GRADIO_CACHE_DIR` environment variable in the script to a writable directory.

5. Update your ChatGroq API key:
   - Replace the placeholder `groq_api_key` in the script with your valid API key.

## Usage

### Running the Application
1. Launch the Gradio interface:
   ```bash
   python app.py
   ```
2. Open the displayed URL in your browser.

### Interacting with the Interface
- **Upload Files**: Upload images or PDFs of documents.
- **Select Document Type**: Choose the type of document (e.g., salary slip).
- **Select Chart Type**: Choose between Pie Chart or Bar Chart for visualization.
- **Dataset Name**: Optionally specify a MongoDB dataset name for image retrieval.

### Output
- **Extracted Data Table**: Displays structured data extracted from the documents.
- **Charts**: Visual comparison of extracted fields.

## Code Structure
- `app.py`: Main script containing the entire pipeline.
- `requirements.txt`: List of dependencies.
- `README.md`: Documentation for the project.

## Key Functions
### OCR and Preprocessing
- `crop_percent`: Crops images based on percentage.
- `convert_pdf_to_images`: Converts PDF pages to images.
- `process_images_with_easyocr`: Processes images with EasyOCR to extract text.

### LLM Integration
- `extract_data_with_llm`: Extracts structured data from OCR text using ChatGroq.
- `get_prompt_for_document`: Generates prompts for specific document types.

### MongoDB Integration
- `retrieve_images_from_mongodb`: Fetches images from MongoDB GridFS.

### Data Visualization
- `create_chart`: Creates Pie or Bar charts for field-wise comparison.
- `convert_figure_to_image`: Converts Matplotlib figures to images for display.

### Gradio Interface
- `gradio_interface`: Main function connecting all components to the Gradio UI.

## Configuration
- **MongoDB URI**: Update the `mongo_uri` variable with your credentials.
- **Gradio Cache Directory**: Modify `GRADIO_CACHE_DIR` to a suitable directory.

## Dependencies
- `easyocr`
- `pymongo`
- `gridfs`
- `pdf2image`
- `langchain`
- `gradio`
- `matplotlib`
- `tabulate`
- `Pillow`

## Known Issues
- Limited support for non-English text.
- High memory usage for large PDFs.
- Ensure proper permissions for Gradio cache directory.

## Future Enhancements
- Add support for additional document types.
- Enhance multilingual OCR capabilities.
- Optimize LLM integration for faster processing.
- Implement error-handling for MongoDB operations.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or contributions, please contact:
- **Name**: [Nandhakumar K P]
- **Email**: [Your Email]

## Acknowledgments
- EasyOCR for OCR capabilities.
- ChatGroq and LLaMA models for advanced text analysis.
- Gradio for providing an easy-to-use interface.
- MongoDB for efficient image storage and retrieval.

# OCR_of_Bank_Statements
