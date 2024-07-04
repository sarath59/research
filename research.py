import os
import warnings
import atexit
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", message="Overriding of current TracerProvider is not allowed")

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Set API keys directly in the code
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
os.environ["SERPER_API_KEY"] = "d4e81a83028357dc5af70321c25760e1dd07e44f"

from crewai_tools import (
    FileReadTool,
    ScrapeWebsiteTool,
    MDXSearchTool,
    SerperDevTool,
    WebsiteSearchTool,
    CodeDocsSearchTool,
    PDFSearchTool
)

pdf_tool = PDFSearchTool()
file_tool = FileReadTool()
mdx_tool = MDXSearchTool()
web_rag_tool = WebsiteSearchTool()
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
code_docs_tool = CodeDocsSearchTool()

# Define the language model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Researcher Agent
researcher = Agent(
    role="PhD level researcher",
    goal="Research about given {topic} and do extensive analysis on given {topic}",
    tools=[scrape_tool, search_tool, pdf_tool, web_rag_tool],
    verbose=True,
    memory=True,
    backstory=(
        "You are a PhD level researcher who is specialized in analyzing the content on the websites, "
        "given data and files and give excellent analysis on the given {topic}. The content you get from all "
        "these resources would be used to create content and also you should give references to every piece of "
        "content that is analyzed and given by you."
    ),
    allow_delegation=True,
    cache=True,
    llm=llm
)

# Writer Agent
writer = Agent(
    role='Writer',
    goal='Compile the gathered information into a structured research paper.',
    verbose=True,
    memory=True,
    backstory='A skilled writer with a knack for transforming data into compelling narratives.',
    tools=[file_tool, search_tool, scrape_tool],
    llm=llm
)

# Reviewer Agent
reviewer = Agent(
    role='Reviewer',
    goal='Review the compiled research paper for accuracy and format it correctly.',
    verbose=True,
    memory=True,
    backstory='An experienced editor with a keen eye for detail.',
    llm=llm
)

# Research Task
research_task = Task(
    description=(
        "Collect detailed information and references on the given topic ({topic}). "
        "Use available tools to gather comprehensive content from various sources. "
        "Ensure to identify and document relevant references."
    ),
    expected_output=(
        "A structured document containing a comprehensive list of references and "
        "summaries of information on {topic}."
    ),
    agent=researcher,
    async_execution=True
)

# Writing Task
writing_task = Task(
    description=(
        "Compile the gathered information into a structured research paper on {topic}. "
        "Ensure the paper includes an abstract, introduction, main content, conclusion, and references. "
        "Format the content according to academic standards."
    ),
    expected_output=(
        "A well-structured research paper on {topic}, formatted with a title, abstract, "
        "table of contents, introduction, main content, conclusion, and references."
    ),
    agent=writer,
    async_execution=True,
    context=[research_task]  # Depends on research task output
)

# Review Task
review_task = Task(
    description=(
        "Review the research paper compiled on {topic} for accuracy, coherence, and formatting. "
        "Make necessary corrections and ensure that the paper adheres to academic standards. "
        "Provide final approval and suggestions for improvement if needed."
    ),
    expected_output=(
        "A final draft of the research paper on {topic}, ready for submission with all corrections "
        "and improvements made."
    ),
    agent=reviewer,
    async_execution=True,
    context=[writing_task]  # Depends on writing task output
)

# Define Crew
research_crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
)

executor = ThreadPoolExecutor()

def shutdown():
    executor.shutdown(wait=True)

atexit.register(shutdown)

def main():
    try:
        user_choice = input("Do you want to (1) write a research paper or (2) test your code? Enter 1 or 2: ")

        if user_choice == "1":
            topic = input("Enter the research topic: ")
            result = research_crew.kickoff(inputs={'topic': topic})
            print(result)
        else:
            print("Invalid choice. Please enter 1.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        shutdown()

if __name__ == "__main__":
    main()
