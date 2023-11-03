import re
import base64
import json
import requests
import os

GLOBAL_DIR_TXT_FOLDER = './helpers/cccd_text_proccesser/txts/'

def txttoarray(file_path):
    prefixes = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Remove leading and trailing whitespace, and convert to lowercase
            prefix = line.strip().lower()
            prefixes.append(prefix)
    return prefixes



def google_cloud_vision_api(image_path, api_key):
    url = 'https://vision.googleapis.com/v1/images:annotate?key=' + api_key
    with open(image_path, 'rb') as image_file:
        content = base64.b64encode(
            image_file.read()).decode('UTF-8')
    # Prepare the request body as a dictionary
    request_body = {
        "requests": [
            {
                "image": {
                    "content": content
                },
                "features": [
                    {
                        "type": "TEXT_DETECTION"
                    }
                ]
            }
        ]
    }
    # Convert the request_body dictionary to a JSON string
    json_data = json.dumps(request_body)

    # Send the POST request
    headers = {
        'Content-Type': 'application/json'}
    response = requests.post(
        url, data=json_data, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if 'responses' in data:
            full_text_annotation = data['responses'][0]['fullTextAnnotation']
            if 'text' in full_text_annotation:
                return full_text_annotation['text']

    print("Error response:", response.text)
    return None

                               
# Name: extract_text_after_prefix
def name_processer(text):
    # Define a list of prefixes you want to look for
    prefixes = txttoarray(f'{GLOBAL_DIR_TXT_FOLDER}name.txt')
    # Initialize the text after prefix
    text_after_prefix = None

    # Try to find a matching prefix and extract the text
    for prefix in prefixes:
        regex_pattern = rf"{re.escape(prefix)}\s*(.*)"
        match = re.search(regex_pattern, text, re.IGNORECASE)

        if match:
            text_after_prefix = match.group(1).strip()
            break

    if text_after_prefix:
        print("Text after prefix:", text_after_prefix)
        return text_after_prefix
    else:
        print("No matching prefix found in the text.")
        return text

def proccess_poo(save_im_dir):
    result_text = google_cloud_vision_api(save_im_dir, os.environ.get('GOOGLE_API_KEY'))
    # result_text =""
    # Define a list of keywords that may precede the desired information
    keywords = txttoarray(f'{GLOBAL_DIR_TXT_FOLDER}poo.txt')

    # Initialize the desired text
    desired_text = None

    # Try to find a keyword and extract the text
    for keyword in keywords:
        match = re.search(
            rf"{keyword}[:\/](.*?)(?=[\n\"]|$)", result_text)
        if match:
            # Remove leading/trailing spaces
            desired_text = match.group(
                1).strip()
            break
    if desired_text:
        print(desired_text)
        return desired_text
    else:
        print("Keywords not found in the text.")
        return result_text
    
def proccess_por(save_im_dir):
    text = google_cloud_vision_api(save_im_dir, os.environ.get('GOOGLE_API_KEY'))
    # text = "I Place of residence Tổ 1, Hiệp Tân\nMỹ Hiệp Sơn, Hòn Đất, Kiên Giang "
    # Define keywords to look for
    keywords = txttoarray(f'{GLOBAL_DIR_TXT_FOLDER}por.txt')
    # Initialize the desired text
    desired_text = None

    # Try to find a keyword and extract the text
    for keyword in keywords:
        if keyword in text:
            desired_text = text.split(
                keyword, 1)[1]
            break

    if desired_text:
        desired_text = desired_text.strip()  # Remove leading/trailing spaces
        print(desired_text)
        # get the value behind :
        desired_text = desired_text.split(':', 1)[1] if ':' in desired_text else desired_text
        # Remove leading/trailing spaces
        desired_text = desired_text.strip()
        return desired_text
    else:
        print("Keywords not found in the text.")
        return text