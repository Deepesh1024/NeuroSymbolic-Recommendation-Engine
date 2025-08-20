# NeuroSymbolic-Recommendation-Engine
This repository is a NeuroSymbolic Recommendation Engine, a system that leverages AI and symbolic reasoning to recommend suitable candidates for job openings based on their skills and experience. The engine extracts skills from job descriptions and matches them with developer profiles, considering both explicit and implied skills.

The repository consists of several Python files, including `ants.py`, `app.py`, and `ns.py`, which contain various components of the engine. The `ants.py` file provides instructions for extracting programming languages, frameworks, tools, and methodologies from job descriptions. The `app.py` file outlines the technical stack and tools used in the engine, including PyTorch, Ray, and Slurm.

The `ns.py` file contains the core logic of the engine, including:

1. **Skill extraction**: Extracting skills from job descriptions using natural language processing techniques.
2. **Developer profiles**: Storing developer profiles with their skills and experience.
3. **Experience level inference**: Inferring the experience level of developers based on their skills and job descriptions.

Example code snippets from the repository include:
```python
# Extract job requirements
job_requirements = system.extract_skills_from_text(job_description)

# Developer profiles
developer_profiles = {
    "Ananya Sharma": ...
}

# Experience level inference
experience_inference = {
    'senior_indicators': [...],
    'intermediate_indicators': [...]
}
```
Overall, this repository provides a comprehensive solution for recommending suitable candidates for job openings based on their skills and experience, leveraging AI and symbolic reasoning techniques.
