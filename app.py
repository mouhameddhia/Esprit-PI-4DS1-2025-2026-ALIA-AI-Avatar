import os
import json
import dotenv
import gradio as gr
from google import genai
dotenv.load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def resume_to_json(filepath):
    uploaded_file = client.files.upload(file=filepath)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            """Extract the following resume into a JSON object with the fields: 
            name, contact information, education, work experience, skills, and certifications. 
            Format the output as a JSON object, answer directly without filler words like 'I think'.
            """,
            uploaded_file,
        ]
    )
    text_output = response.text
    # locate the first "{" and the last "}"
    json_start = text_output.find("{")
    json_end = text_output.rfind("}") + 1
    return json.loads(text_output[json_start:json_end])


demo = gr.Interface(
    fn=resume_to_json,
    inputs="file",
    outputs="json",
    title="Resume to JSON Extractor",
    api_name="resume_to_json_extractor",
)

demo.launch()
