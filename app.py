import os
import json
import dotenv
import gradio as gr
from google import genai
dotenv.load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

with open("./cv_template_camelCase.json", "r") as f:
    CV_TEMPLATE = f.read() # read as text and not json to pass to the model

def resume_to_json(filepath):
    uploaded_file = client.files.upload(file=filepath)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            f"""You will receive a resume as the next input. Convert it into a single JSON object that exactly matches the template below:
            ```json
            {CV_TEMPLATE}
            ```
            Output requirements:
            - Return only one valid JSON object and nothing else (no explanations, no headings, no surrounding text or code fences).
            - Preserve the template's keys and structure exactly.
            - For any missing or unknown value, use the template's empty defaults: empty string "" for text fields and empty list [] for list fields.
            - For list fields (e.g. skills, languages, certifications) return arrays of short strings.
            - For workExperience and education entries, populate subfields (title, company, startDate, endDate, description); if a subfield is missing use an empty string.
            - Ensure the output is valid, parseable JSON.

            Do not include any additional text before or after the JSON object.
            """,
            uploaded_file,
        ]
    )
    text_output = response.text
    # locate the first "{" and the last "}"
    json_start = text_output.find("{")
    json_end = text_output.rfind("}") + 1
    return json.loads(text_output[json_start:json_end])


def hello(name):
    return f"Hello {name}!"

resume_to_json_extractor = gr.Interface(
    fn=resume_to_json,
    inputs="file",
    outputs="json",
    title="Resume to JSON Extractor",
    api_name="resume_to_json_extractor",
)

hello_interface = gr.Interface(
    fn=hello,
    inputs="text",
    outputs="text",
    title="Hello Interface",
    api_name="hello_interface",
)

demo = gr.TabbedInterface(
    [resume_to_json_extractor, hello_interface],
    ["Resume to JSON", "Hello"]
)

demo.launch()
