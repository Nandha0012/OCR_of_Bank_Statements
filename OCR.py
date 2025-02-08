import os
import easyocr
import re
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pdf2image import convert_from_path
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import gradio as gr
from io import BytesIO
from pymongo import MongoClient
from gridfs import GridFS
import urllib.parse

# Set custom Gradio cache directory to avoid permission issues
os.environ["GRADIO_CACHE_DIR"] = "C:\\Temp\\gradio_cache"  # Adjust this path as needed

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.1-70b-versatile"
)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)  # Use CPU

# MongoDB Configuration (Escaped username and password)
username = urllib.parse.quote_plus("admin")
password = urllib.parse.quote_plus("admin123@BFSI")
mongo_uri = f"mongodb+srv://{username}:{password}@bfsi.dyb8u.mongodb.net/?retryWrites=true&w=majority&appName=BFSI"
db_name = 'bfsi'

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client[db_name]
fs = GridFS(db)

# Define cropping limits
CROP_LIMITS = {"upper_percent": 0.00, "lower_percent": 0.00}

def ensure_directory(directory):
    os.makedirs(directory, exist_ok=True)

def crop_percent(image_path, output_path, upper_percent=0.00, lower_percent=0.00):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            crop_upper = height * upper_percent
            crop_lower = height * (1 - lower_percent)
            cropped_img = img.crop((0, crop_upper, width, crop_lower))

            if cropped_img.mode != 'RGB':
                cropped_img = cropped_img.convert('RGB')

            cropped_img.save(output_path, format="JPEG")
    except Exception as e:
        print(f"Error cropping {image_path}: {e}")

def convert_pdf_to_images(pdf_path, output_folder):
    ensure_directory(output_folder)
    try:
        pages = convert_from_path(pdf_path, dpi=300)
        image_paths = []
        for i, page in enumerate(pages):
            image_path = os.path.join(output_folder, f"{os.path.basename(pdf_path)}page{i + 1}.jpg")
            page.save(image_path, "JPEG")
            image_paths.append(image_path)
        return image_paths
    except Exception as e:
        print(f"Error converting {pdf_path} to images: {e}")
        return []

def process_images_with_easyocr(image_paths):
    text_list = []
    for image_path in image_paths:
        try:
            if CROP_LIMITS:
                cropped_path = os.path.join(os.path.dirname(image_path), f"cropped_{os.path.basename(image_path)}")
                crop_percent(image_path, cropped_path, CROP_LIMITS.get("upper_percent"), CROP_LIMITS.get("lower_percent"))
                image_path = cropped_path
            ocr_results = reader.readtext(image_path)
            recognized_text = " ".join([text for _, text, _ in ocr_results])
            text_list.append((image_path, recognized_text))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    return text_list

def get_prompt_for_document(input_type, ocr_text):
    prompts = {
        "salary slip": "Extract the following from the salary slip: gross salary, house rent allowances, conveyance allowances, net salary, basic amount. Text: {{ocr_text}}",
        "balance slip": "Extract the following from the balance slip: account holder name, account number, balance, date, bank name. Text: {{ocr_text}}",
        "cash slip": "Extract the following from the cash slip: transaction date, amount, transaction ID, bank/ATM ID. Text: {{ocr_text}}"
    }
    return prompts[input_type].replace("{{ocr_text}}", ocr_text)

def extract_data_with_llm(ocr_text, input_type):
    try:
        prompt_text = get_prompt_for_document(input_type, ocr_text)
        prompt = PromptTemplate(input_variables=["ocr_text"], template=prompt_text)
        result = prompt | llm
        response = result.invoke({"ocr_text": ocr_text})
        return response.content
    except Exception as e:
        print(f"Error interacting with LLM: {e}")
        return {"error": str(e)}

def retrieve_images_from_mongodb(dataset_name, document_type):
    try:
        query = {"metadata.dataset_name": dataset_name, "metadata.folder_name": {"$regex": document_type, "$options": 'i'}}
        files = list(fs.find(query).limit(2))

        if not files:
            return "No images found in MongoDB for the specified query."

        image_paths = []
        for file in files:
            file_path = os.path.join("retrieved_images", file.filename)
            with open(file_path, 'wb') as f:
                f.write(file.read())
            image_paths.append(file_path)

        return image_paths
    except Exception as e:
        print(f"Error retrieving images from MongoDB: {e}")
        return []

def process_user_request(files, input_type, dataset_name):
    extracted_data = {}

    if files:
        image_paths = []
        for file in files:
            if isinstance(file, str) and file.endswith('.pdf'):
                image_paths += convert_pdf_to_images(file, "converted_images")
            elif isinstance(file, BytesIO):
                file_path = os.path.join("uploaded_images", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.read())
                image_paths.append(file_path)
            else:
                image_paths.append(file)

        ocr_data = process_images_with_easyocr(image_paths)

        for img_path, ocr_text in ocr_data:
            extracted_data[img_path] = extract_data_with_llm(ocr_text, input_type)

    if dataset_name:
        mongodb_images = retrieve_images_from_mongodb(dataset_name, input_type)
        if isinstance(mongodb_images, list):
            extracted_data.update(process_user_request(mongodb_images, input_type, ""))

    return extracted_data

def field_wise_comparison_data(data):
    field_data = {}
    for image, attributes in data.items():
        if isinstance(attributes, str):
            matches = re.findall(r"(\*\*\w+(?: \w+)*\*\*):\s*([^\n]+)", attributes)
            for key, value in matches:
                if key not in field_data:
                    field_data[key] = []
                field_data[key].append(value)
        elif isinstance(attributes, dict):
            for key, value in attributes.items():
                if key not in field_data:
                    field_data[key] = []
                field_data[key].append(value)
    return field_data

def create_chart(data, chart_type, field_names, image_name):
    try:
        if not data:
            print(f"No data available for fields: {field_names}")
            return None

        # Extract values for the specified fields
        field_values = [data.get(field, 'N/A') for field in field_names]

        if not any(field_values):
            print(f"No values found for the selected fields: {field_names}")
            return None

        # Create chart
        fig, ax = plt.subplots(figsize=(6, 6))
        if chart_type == "Pie Chart":
            ax.pie(field_values, labels=field_names, autopct="%1.1f%%", startangle=90)
            ax.set_title(f"Field-wise Comparison for {image_name} (Pie Chart)")
        elif chart_type == "Bar Chart":
            ax.bar(field_names, field_values)
            ax.set_title(f"Field-wise Comparison for {image_name} (Bar Chart)")
            ax.set_xticks(range(len(field_names)))
            ax.set_xticklabels(field_names, rotation=45)

        return fig
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

def convert_figure_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)

def format_extracted_table(data):
    table_html = ""
    for image, details in data.items():
        formatted_details = ""
        if isinstance(details, str):
            matches = re.findall(r"(\*\*\w+(?: \w+)*\*\*):\s*([^\n]+)", details)
            for field, value in matches:
                formatted_details += f"<tr><td>{field}</td><td>{value}</td></tr>"

        table_html += f"<h3>{image}</h3><table style='width:100%; border-collapse:collapse; border:1px solid black;'>"
        table_html += "<thead><tr style='background-color:#f2f2f2;'><th>Field Name</th><th>Field Value</th></tr></thead>"
        table_html += f"<tbody>{formatted_details}</tbody></table><br>"

    return table_html

def gradio_interface(files, input_type, chart_type, dataset_name):
    extracted_data = {}
    extracted_data.update(process_user_request(files, input_type, dataset_name))

    chart_images = []

    # Create charts for each image's extracted data
    for image_path, data in extracted_data.items():
        field_names = []
        example_data = {}

        # Define field names and example data based on the document type
        if input_type == "salary slip":
            field_names = [
                "Gross Salary", 
                "House Rent Allowances", 
                "Conveyance Allowances", 
                "Net Salary", 
                "Basic Amount"
            ]
            example_data = {
                "Gross Salary": 20000,
                "House Rent Allowances": 3600,
                "Conveyance Allowances": 1600,
                "Net Salary": 20000,
                "Basic Amount": 15000
            }

        elif input_type == "cash slip":
            field_names = [
                "Amount",
                "Transaction Date",
                "Transaction ID"
            ]
            example_data = {
                "Amount": 12345678,
                "Transaction Date": "2025-01-03",
                "Transaction ID": "TXN123456789"
            }

        elif input_type == "balance slip":
            field_names = [
                "Account Holder Name",
                "Account Number",
                "Balance",
                "Date",
                "Bank Name"
            ]
            example_data = {
                "Account Holder Name": "John Doe",
                "Account Number": "123456789",
                "Balance": 5000,
                "Date": "2025-01-03",
                "Bank Name": "ABC Bank"
            }

        # Create a chart for the extracted data from this image
        fig = create_chart(example_data, chart_type, field_names, os.path.basename(image_path))
        if fig:
            chart_images.append(convert_figure_to_image(fig))

    # Format the extracted data as an HTML table for all document types
    formatted_table = format_extracted_table(extracted_data)

    return formatted_table, chart_images

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[ 
        gr.File(file_types=[".jpg", ".png", ".jpeg", ".pdf"], label="Upload Files", file_count="multiple"),
        gr.Radio(["salary slip", "balance slip", "cash slip"], label="Select Document Type"),
        gr.Radio(["Pie Chart", "Bar Chart"], label="Select Chart Type"),
        gr.Textbox(label="Dataset Name (optional)", placeholder="Enter dataset name for MongoDB (optional)")
    ],
    outputs=[
        gr.HTML(label="Extracted Data Table"),
        gr.Gallery(label="Charts", columns=1)  # Multiple charts will be shown in the gallery
    ],
    title="OCR Data Extraction and Visualization",
    description="<div style='font-size: 16px; color: #4A90E2; text-align: center;'>Upload documents or use MongoDB dataset to extract specific data using OCR and LLM. Visualize field-wise comparison charts.</div>",
    theme="compact",
    css="""
        body { background-color: #F8F9FA; font-family: 'Arial', sans-serif; }
        .output-container { background-color: #FFFFFF; padding: 15px; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); }
        .gradio-container { padding: 20px; }
    """
)

iface.launch()
