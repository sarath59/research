from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import warnings
import traceback
import atexit
from concurrent.futures import ThreadPoolExecutor
import json
import logging

warnings.filterwarnings("ignore", message="Overriding of current TracerProvider is not allowed")

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Set API keys (use environment variables in production)
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

from crewai_tools import CodeDocsSearchTool

code_docs_tool = CodeDocsSearchTool()

# Define the language model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Idea Analyst Agent
idea_analyst = Agent(
    role="Senior Idea Analyst",
    goal="Understand and expand upon the essence of ideas, making sure they are great and focus on real pain points others could benefit from.",
    backstory=(
        "Recognized as a thought leader, I thrive on refining concepts into campaigns that resonate with audiences. "
        "My expertise lies in analyzing raw data and transforming it into compelling narratives that highlight an individual's unique value proposition. "
        "I have a keen understanding of various industries and can tailor content to specific target audiences, ensuring that each portfolio tells a captivating story."
    ),
    verbose=True,
    llm=llm
)

# Communications Strategist Agent
communications_strategist = Agent(
    role="Senior Communications Strategist",
    goal="Craft compelling stories using the Golden Circle method to captivate and engage people around an idea.",
    backstory=(
        "A narrative craftsman for top-tier launches, I reveal the why behind projects, aligning with visions and speaking to audiences. "
        "My approach involves deep diving into an individual's background, skills, and projects to create a cohesive narrative that resonates with potential employers or clients. "
        "I specialize in creating content that not only informs but also inspires and motivates the target audience to take action."
    ),
    verbose=True,
    llm=llm
)

# React Engineer Agent
react_engineer = Agent(
    role="Senior React Engineer",
    goal="Build an intuitive, aesthetically pleasing, and high-converting landing page.",
    backstory=(
        "A coding virtuoso and design enthusiast, expert in Tailwind, you're known for crafting beautiful websites that provide seamless user experiences. "
        "You have been tasked to build a portfolio page for a user who is an immigrant from India, came to the USA for their master's in computer science, and started an AI company. "
        "The portfolio should be modern, visually appealing, and user-friendly. The design should incorporate the color scheme and design preferences provided by the user. "
        "Your expertise includes creating responsive designs, implementing modern web animations, and ensuring cross-browser compatibility. "
        "You have a track record of developing high-performance websites that not only look great but also achieve business objectives."
    ),
    verbose=True,
    allow_code_execution=True,
    llm=llm
)

# Content Editor Agent
content_editor = Agent(
    role="Senior Content Editor",
    goal="Ensure the landing page content is clear, concise, and captivating.",
    backstory=(
        "With a keen eye for detail and a passion for storytelling, you have refined content for leading brands, turning bland text into engaging stories. "
        "Your expertise lies in optimizing content for both user engagement and search engine visibility. "
        "You excel at crafting headlines, call-to-actions, and microcopy that drive user interaction and conversion. "
        "Your editing skills ensure that the final content is not only grammatically perfect but also aligns with the user's personal brand and professional goals."
    ),
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# Task to gather user data and expand upon ideas
gather_user_data_task = Task(
    description=(
        "Analyze the provided user data including name, GitHub profile, project names, skills, color preferences, and bio. "
        "Expand upon these ideas to ensure the landing page content is compelling and addresses real pain points others could benefit from. "
        "Consider the user's background as an immigrant from India who came to the USA for a master's in computer science and started an AI company. "
        "Identify key themes and unique selling points in the user's story that can be highlighted in the portfolio. "
        "Provide a detailed analysis of how each piece of information can be leveraged to create a standout portfolio."
    ),
    expected_output=(
        "A structured document containing:\n"
        "1. Expanded user data analysis\n"
        "2. Key themes identified in the user's story\n"
        "3. Unique selling points to highlight\n"
        "4. Suggestions for leveraging each piece of information in the portfolio"
    ),
    agent=idea_analyst,
    async_execution=False
)

# Task to craft the story and overall message
craft_story_task = Task(
    description=(
        "Using the expanded user data and analysis, craft a compelling story and overall message for the landing page. "
        "Apply the Golden Circle method to articulate the 'why', 'how', and 'what' of the user's professional journey. "
        "Incorporate the user's background as an immigrant, their education in the USA, and their entrepreneurial venture in AI. "
        "Ensure the message is engaging, aligns with the user's goals and preferences, and resonates with potential employers or clients in the tech industry. "
        "Create a narrative that ties together the user's skills, projects, and experiences into a cohesive personal brand story."
    ),
    expected_output=(
        "A document containing:\n"
        "1. The overarching story and message for the landing page\n"
        "2. Key points addressing the 'why', 'how', and 'what' of the user's journey\n"
        "3. A outline of how the narrative will be presented across different sections of the portfolio\n"
        "4. Suggested taglines or headlines that capture the essence of the user's personal brand"
    ),
    agent=communications_strategist,
    async_execution=False,
    context=[gather_user_data_task]
)

# Task to build the landing page
build_landing_page_task = Task(
    description=(
        "Build the landing page using HTML, CSS, and JavaScript, incorporating the user's data, crafted story, and design preferences. "
        "Ensure the design is modern, visually appealing, and user-friendly. The landing page should have the following features:\n"
        "1. Responsive design that works well on desktop, tablet, and mobile devices\n"
        "2. A hero section featuring the user's name, tagline, and a call-to-action\n"
        "3. An 'About Me' section that tells the user's story, including their journey from India to the USA\n"
        "4. A skills section showcasing the user's technical abilities\n"
        "5. A projects section highlighting the user's key works, including their AI company\n"
        "6. A contact section with a form and links to the user's GitHub and other relevant profiles\n"
        "7. Smooth scrolling and subtle animations to enhance user experience\n"
        "8. Incorporation of the user's color preferences in the design\n"
        "9. Optimized performance with minimal load times\n"
        "The landing page should be a single file with inline CSS and JavaScript for easy deployment."
    ),
    expected_output=(
        "A single HTML file which includes:\n"
        "1. Semantic HTML structure\n"
        "2. Inline CSS with styles reflecting the user's preferences\n"
        "3. Inline JavaScript for interactivity and animations\n"
        "4. All content sections as described, populated with the user's data and crafted story\n"
        "5. Responsive design implementations\n"
        "6. Performance optimizations"
    ),
    agent=react_engineer,
    async_execution=False,
    allow_code_execution=True,
    context=[gather_user_data_task, craft_story_task]
)

# Task to refine and perfect the content
refine_content_task = Task(
    description=(
        "Review and refine the content of the landing page to ensure it is clear, concise, and captivating. "
        "Pay special attention to:\n"
        "1. Headlines and subheadings - ensure they are attention-grabbing and informative\n"
        "2. Body text - optimize for readability and engagement\n"
        "3. Call-to-actions - make them compelling and aligned with the user's goals\n"
        "4. Project descriptions - highlight the impact and technologies used\n"
        "5. Skills presentation - ensure they are presented in a way that showcases the user's expertise\n"
        "6. Overall narrative flow - make sure the user's story is cohesive from start to finish\n"
        "7. SEO optimization - include relevant keywords naturally in the content\n"
        "8. Tone and voice - ensure it matches the user's personal brand and target audience\n"
        "Make necessary edits to the HTML file to implement these content improvements."
    ),
    expected_output=(
        "An updated HTML file with refined content, including:\n"
        "1. Improved headlines and subheadings\n"
        "2. Optimized body text\n"
        "3. Enhanced call-to-actions\n"
        "4. Polished project descriptions\n"
        "5. Well-presented skills section\n"
        "6. Cohesive narrative throughout the page\n"
        "7. SEO-friendly content\n"
        "8. Consistent tone and voice"
    ),
    agent=content_editor,
    async_execution=False,
    context=[gather_user_data_task, craft_story_task, build_landing_page_task]
)

# Define Crew
landing_page_crew = Crew(
    agents=[idea_analyst, communications_strategist, react_engineer, content_editor],
    tasks=[gather_user_data_task, craft_story_task, build_landing_page_task, refine_content_task],
    process=Process.sequential,
)

executor = ThreadPoolExecutor()

def shutdown():
    executor.shutdown(wait=True)

atexit.register(shutdown)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/generate', methods=['POST'])
def generate_portfolio():
    try:
        user_data = request.json
        logger.info(f"Received user data: {user_data}")
        
        result = landing_page_crew.kickoff(inputs=user_data)
        logger.info(f"Crew result: {result}")
        
        parsed_result = parse_crew_result(result)
        logger.info(f"Parsed result: {parsed_result}")
        
        filename = generate_html_file(parsed_result, user_data['name'])
        
        return jsonify({"result": parsed_result, "file": filename})
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

def parse_crew_result(result):
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract HTML content
            html_start = result.find('<!DOCTYPE html>')
            html_end = result.rfind('</html>') + 7
            if html_start != -1 and html_end != -1:
                return {"content": result[html_start:html_end]}
            else:
                return {"content": result}
    elif isinstance(result, dict):
        return result
    else:
        raise ValueError("Unexpected result format")

def generate_html_file(parsed_result, name):
    html_content = parsed_result.get('content', '')
    
    filename = f"portfolio_{name.lower().replace(' ', '_')}.html"
    with open(filename, 'w') as f:
        f.write(html_content)
    
    return filename

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('.', filename)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)