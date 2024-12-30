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
from tabulate import tabulate

# Set custom Gradio cache directory to avoid permission issues
os.environ["GRADIO_CACHE_DIR"] = "C:\\Temp\\gradio_cache"  # Adjust this path as needed

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=" ",
    model_name="llama-3.1-70b-versatile"
)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)  # Use CPU

# MongoDB Configuration (Escaped username and password)
username = urllib.parse.quote_plus(" ")
password = urllib.parse.quote_plus(" ")
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
            elif hasattr(file, 'name'):
                image_paths.append(file.name)
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
            matches = re.findall(r"(\w+(?: \w+)*):\s*([^\n]+)", attributes)
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

def create_chart(data, chart_type, field):
    try:
        if not data:
            print(f"No data available for field: {field}")
            return None

        aggregated_values = [item for sublist in data for item in sublist]
        unique_values = list(set(aggregated_values))
        counts = [aggregated_values.count(value) for value in unique_values]
        colors = plt.cm.Paired.colors[:len(unique_values)]

        fig, ax = plt.subplots(figsize=(6, 6))

        if chart_type == "Pie Chart":
            ax.pie(counts, labels=unique_values, autopct="%1.1f%%", startangle=90, colors=colors)
            ax.set_title(f"{field} Comparison (Pie Chart)")

        elif chart_type == "Bar Chart":
            ax.bar(unique_values, counts, color=colors)
            ax.set_title(f"{field} Comparison (Bar Chart)")
            ax.set_xticks(range(len(unique_values)))
            ax.set_xticklabels(unique_values, rotation=45)

        return fig
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

def convert_figure_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)

def gradio_interface(files, input_type, chart_type, dataset_name):
    extracted_data = {}
    extracted_data.update(process_user_request(files, input_type, dataset_name))

    field_data = field_wise_comparison_data(extracted_data)
    charts = []
    for field, values in field_data.items():
        fig = create_chart(values, chart_type, field)
        if fig:
            charts.append(convert_figure_to_image(fig))

    table_data = []
    for img, data in extracted_data.items():
        if isinstance(data, str):
            table_data.append([img, data])
        elif isinstance(data, dict):
            for key, value in data.items():
                table_data.append([img, key, value])

    result_table = tabulate(table_data, headers=["Image", "Field", "Value"], tablefmt="grid")

    return result_table, charts

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[ 
        gr.File(file_types=[".jpg", ".png", ".jpeg", ".pdf"], label="Upload Files", file_count="multiple"),
        gr.Radio(["salary slip", "balance slip", "cash slip"], label="Select Document Type"),
        gr.Radio(["Pie Chart", "Bar Chart"], label="Select Chart Type"),
        gr.Textbox(label="Dataset Name (optional)", placeholder="Enter dataset name for MongoDB (optional)")
    ],
    outputs=[
        gr.Textbox(label="Extracted Data Table", lines=20, interactive=False, elem_id="table_output"),
        gr.Gallery(label="Charts", columns=1)  # Charts displayed after the table
    ],
    title="OCR Data Extraction and Visualization",
    description="Upload documents or use MongoDB dataset to extract specific data using OCR and LLM. Visualize field-wise comparison charts."
)

if __name__ == "__main__":
    iface.launch()
