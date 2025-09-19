from http.server import BaseHTTPRequestHandler
import json
import os
from langchain_mistralai.chat_models import ChatMistralAI
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SolutionModel(BaseModel):
    key_points: str = Field(description="Key issues in the described problem that need to be addressed.")
    type_problem: str = Field(description="Type of problem described in the prompt. Whether it is a personal, professional, technical, or other type of problem.")
    fault: Optional[str] = Field(default=None, description="Only if it is a personal problem. Consider the speaker/prompter as 1 person, if more than one person is involved, describe the fault of each person without any bias and mention who was more at fault.")
    solution: str = Field(description="A detailed solution to the problem described in the prompt. without bias and/or sugarcoating.")

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        
        user_input = data.get('message', '')
        
        if not user_input:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = json.dumps({'error': 'No message provided'})
            self.wfile.write(response.encode())
            return
            
        try:
            # Initialize the model (with error handling)
            model = ChatMistralAI(
                model="magistral-small-2507", 
                temperature=0.8,
                mistral_api_key=os.environ.get('MISTRAL_API_KEY')
            )
            
            struct_model = model.with_structured_output(SolutionModel, method="json_schema")
            result = struct_model.invoke(user_input)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response_data = {
                'key_points': result.key_points,
                'type_problem': result.type_problem,
                'fault': result.fault,
                'solution': result.solution
            }
            
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = json.dumps({'error': str(e)})
            self.wfile.write(response.encode())