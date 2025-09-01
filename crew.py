import os
from crewai import Agent, Crew, Task, Process
from crewai_tools import SerperDevTool
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))
model_name = os.getenv("MODEL", "gemini/gemini-1.5-flash")

def validate_chapters(output: str, min_chapters=8, max_chapters=10, min_words=300, max_words=400):
    chapters = [c for c in output.split("Chapter") if c.strip()]
    if len(chapters) < min_chapters or len(chapters) > max_chapters:
        return f"❌ Found {len(chapters)} chapters. Please rewrite with {min_chapters}-{max_chapters} chapters."
    
    for i, chapter in enumerate(chapters, 1):
        word_count = len(chapter.split())
        if word_count < min_words or word_count > max_words:
            return f"❌ Chapter {i} has {word_count} words. Please rewrite with {min_words}-{max_words} words."
    return output

# ========== AGENTS DEFINITION ==========
ideaCuratorAgent = Agent(
    role="Idea Curator",
    goal="Propose unique book ideas based on user input",
    backstory=(
        "You are a creative thinker who researches book niches, "
        "identifies unique angles, and proposes marketable ideas."
    ),
    allow_delegation=False,
    llm=model_name,
    tools=[search_tool]
)

outlineArchitectAgent = Agent(
    role="Outline Architect",
    goal="Expand chosen idea into a structured book outline",
    backstory="You are skilled at structuring content into logical chapters.",
    llm=model_name
)

chapterWriterAgent = Agent(
    role="Chapter Writer",
    goal="Write detailed drafts for each chapter",
    backstory="You are an articulate writer who adapts tone to fit the audience.",
    llm=model_name
)

editorPolisherAgent = Agent(
    role="Editor/Polisher",
    goal="Refine the draft for clarity, grammar, and consistency",
    backstory="You ensure flow and consistency across the manuscript.",
    llm=model_name
)

formatterAgent = Agent(
    role="Formatter",
    goal="Prepare the final book as a single Markdown document",
    backstory="You convert manuscripts into clean, publish-ready Markdown.",
    llm=model_name
)

# ========== TASKS DEFINITION ==========
generateIdeasTask = Task(
    description="Generate 3 unique book ideas based on 'Frontend Development Masterclass for Developers'.",
    agent=ideaCuratorAgent,
    expected_output="A list of 3 unique and marketable book ideas."
)

createOutlineTask = Task(
    description=(
        "Expand chosen idea into a structured book outline (8-10 chapters)."
        "'Mastering Landing Pages for Frontend Developers'. "
        "The outline must include exactly 4 to 5 chapters, "
        "with clear chapter titles and a short summary for each."
    ),
    agent=outlineArchitectAgent,
    context=[generateIdeasTask],
    expected_output="A detailed outline with 8-10 chapters."
)

draftChaptersTask = Task(
    description=(
        "Write detailed drafts for each chapter."
        "Each chapter must be between 300 and 400 words. "
        "Do not skip any chapters. "
        "Label each chapter clearly (e.g., 'Chapter 1: ...')."
    ),
    agent=chapterWriterAgent,
    context=[createOutlineTask],
    expected_output=(
        "Draft text for each chapter of the book."
        "Each chapter is clearly labeled and contains 300–400 words."
    ),
    output_validator=validate_chapters
)

polishBookTask = Task(
    description="Polish drafts for readability, grammar, and flow.",
    agent=editorPolisherAgent,
    context=[draftChaptersTask],
    expected_output="A polished manuscript with consistent flow."
)

formatBookTask = Task(
    description=(
        "Take the polished manuscript and format it into a single Markdown document. "
        "Do not output multiple HTML files or EPUB markup. "
        "Instead, produce one `.md` file where each chapter starts with an H1 heading (`# Chapter X: Title`). "
        "Within each chapter, use H2/H3 headings if needed, and keep paragraphs as plain text. "
        "The final output must be valid Markdown only."
    ),
    agent=formatterAgent,
    context=[polishBookTask],
    expected_output=(
        "A single Markdown file containing the full book. "
        "Each chapter is clearly separated using `# Chapter N: Title` headings."
    )
)

agents = [
    ideaCuratorAgent,
    outlineArchitectAgent,
    chapterWriterAgent,
    editorPolisherAgent,
    formatterAgent
]

tasks = [
    generateIdeasTask,
    createOutlineTask,
    draftChaptersTask,
    polishBookTask,
    formatBookTask
]

# Create crew
crew = Crew(
    agents=agents,
    tasks=tasks,
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("\n✅ Running the Crew!\n")
    result = crew.kickoff()
    print("\n✅ Final Output:\n", result)
