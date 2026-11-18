#!/usr/bin/python3

import json
from deep_consultation.core import consult_with_deepchat

SYSTEM_PROMPT_2 = """
You are an expert scientific writer specialized in Q1 Computer Science journals
(e.g., IEEE TPAMI, CVPR, ICCV, NeurIPS, Elsevier Pattern Recognition).

Your task is to generate only 1 next paragraph of a research paper.

"""

USER_PROMPT_2 = """
Strict rules:
* Use only supplied information. Do not invent facts, results, methods, or citations. 
* The tone must match a top-tier Q1 journal: formal, precise, and concise.
* No bullet points. No section headers. Only well-structured academic paragraph.
* Do NOT describe the JSON or its structure in the output.
* Do not use exactly the same text sent in JSON, try to use your own words to avoid plagiarism problems.
* Avoid writing texts with redundant information.
* Generate only the text, in English, as LaTeX-ready academic prose.
* Generate the text prioritizing clarity, trying to keep it as short as possible.
"""


def _format_prompt_data(fields):
    """Render selected data as readable labeled text, without JSON braces."""
    lines = []
    for label, value in fields:
        if value is None or value == "" or value == [] or value == {}:
            continue
        if isinstance(value, (list, tuple)):
            lines.append(f"{label}:")
            for item in value:
                if item is None or item == "":
                    continue
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    nested_label, nested_value = item
                    rendered = _format_prompt_data([(nested_label, nested_value)])
                    lines.append(rendered)
                else:
                    lines.append(f"- {item}")
        elif isinstance(value, dict):
            lines.append(f"{label}:")
            for key, item in value.items():
                if item is None or item == "" or item == [] or item == {}:
                    continue
                if isinstance(item, (list, tuple)):
                    lines.append(f"{key}:")
                    lines.extend(f"- {entry}" for entry in item if entry is not None and entry != "")
                else:
                    lines.append(f"{key}: {item}")
        else:
            lines.append(f"{label}: {value}")
    return "\n".join(lines) if lines else "No content was provided for this step."


def _step(prompt, fields):
    """Build one (instruction, selected-data) item for iterative generation."""
    return prompt.strip(), _format_prompt_data(fields)


def generate_initial_context(data):
    """Return concise article metadata shared by every generated paragraph."""
    profile = data.get("paper_profile", {})
    return _format_prompt_data([
        ("Article title", profile.get("title")),
        ("Research domain of the paper", profile.get("domain")),
        ("Target scientific journal", profile.get("target_journal")),
        ("Keywords", profile.get("keywords")),
        ("Writing guidelines", data.get("writing_guidelines")),
    ])


def generate_llm_introduction_commands(data):
    """Return Introduction steps and matching LaTeX-comment paragraph titles."""
    profile = data.get("paper_profile", {})
    problem = data.get("research_problem", {})
    synthesis = data.get("related_work", {}).get("human_curated_synthesis", {})
    contributions = data.get("contributions", [])

    commands = [
        # Context and domain importanc
        _step(
            "Write the opening paragraph of the Introduction. Establish the broad research context and explain its importance. Use the author's summary only to guide emphasis. Return exactly one cohesive paragraph, with no heading and no citations unless citation data is supplied.",
            [("Author's intended summary", profile.get("author_intended_summary")),
             ("Research domain overview", problem.get("research_domain_overview"))],
        ),
        # Specific problem and challenges
        _step(
            "Write the next Introduction paragraph. Narrow the discussion from the broad domain to the specific problem addressed by the paper, then explain its practical or scientific challenges. Do not repeat the opening paragraph. Return exactly one paragraph.",
            [("Author's intended summary", profile.get("author_intended_summary")),
             ("Specific research problem", problem.get("specific_problem")),
             ("Practical challenges", problem.get("practical_challenges"))],
        ),
        # Limitations of existing solutions
        _step(
            "Write the next Introduction paragraph explaining why existing solutions are insufficient. Present the supplied assessment as the motivation for further research; do not make unsupported claims or repeat the problem statement. Return exactly one paragraph.",
            [("Why existing solutions are insufficient", problem.get("why_existing_solutions_are_insufficient"))],
        ),
        # Trends and open problems
        _step(
            "Write the next Introduction paragraph synthesizing shared trends and unresolved problems in the literature. Do not list papers or invent citations; detailed paper-by-paper discussion belongs in the Related Work section. Return exactly one paragraph.",
            [("Common trends in related work", synthesis.get("common_trends")),
             ("Open problems in related work", synthesis.get("open_problems"))],
        ),
        # Research gap
        _step(
            "Write the next Introduction paragraph stating the explicit research gap and explaining the paper's broad response to it. Use the author's summary to keep the framing faithful to the intended work. Do not enumerate contributions yet. Return exactly one paragraph.",
            [("Explicit research gap", synthesis.get("explicit_research_gap")),
             ("Author's intended summary", profile.get("author_intended_summary"))],
        ),
        # Proposed response and contributions
        _step(
            "Write the final paragraph of the Introduction. Present how the paper responds to the research gap and integrate the contributions naturally into prose, without a bullet list. Do not invent implementation details or results. Return exactly one paragraph.",
            [("Author's intended summary", profile.get("author_intended_summary")),
             ("Explicit research gap", synthesis.get("explicit_research_gap")),
             ("Paper contributions", contributions)],
        ),
    ]
    titles = [
        "Context and domain importance",
        "Specific problem and challenges",
        "Limitations of existing solutions",
        "Trends and open problems",
        "Research gap",
        "Proposed response and contributions",
    ]
    return commands, titles


def generate_llm_relatedworks_commands(data):
    """Return Related Work steps and matching LaTeX-comment paragraph titles."""
    related_work = data.get("related_work", {})
    references = related_work.get("references", {})
    synthesis = related_work.get("human_curated_synthesis", {})

    role_groups = {
        str(reference.get("introduction_paragraph_role", "")).strip()
        for reference in references.values()
        if str(reference.get("introduction_paragraph_role", "")).strip()
    }
    commands = []
    titles = []
    included_keys = set()

    for role_value in sorted(role_groups, key=str.casefold):
        role_label = role_value.replace("_", " ").strip().title()
        matching = [
            (key, ref) for key, ref in references.items()
            if str(ref.get("introduction_paragraph_role", "")).strip() == role_value
        ]
        if not matching:
            continue
        included_keys.update(key for key, _ in matching)
        ref_fields = []
        for key, ref in matching:
            ref_fields.append((f"Reference key: {key}", {
                "BibTeX entry (use its citation key for LaTeX citations)": ref.get("bibtex"),
                "Abstract": ref.get("abstract"),
                "Methodological category": ref.get("methodological_category"),
                "Central technical idea": ref.get("central_technical_idea"),
                "Author-reported strengths": ref.get("author_reported_strengths"),
                "Reported limitations": ref.get("reported_limitations"),
                "Relevance to our work": ref.get("relevance_to_our_work"),
                "Assigned literature role": ref.get("introduction_paragraph_role"),
            }))
        commands.append(_step(
            f"Write one Related Work paragraph discussing the supplied {role_label.lower()}. Compare and connect the works around their technical ideas rather than writing a sequence of disconnected summaries. Cite each discussed work using a valid LaTeX \\cite{{key}} command whose key appears in that work's BibTeX entry. Do not invent citation keys or claims. Return exactly one paragraph, with no heading.",
            [(f"{role_label} references", ref_fields)],
        ))
        titles.append(role_label)

    other_refs = [(key, ref) for key, ref in references.items() if key not in included_keys]
    if other_refs:
        ref_fields = []
        for key, ref in other_refs:
            ref_fields.append((f"Reference key: {key}", {
                "BibTeX entry (use its citation key for LaTeX citations)": ref.get("bibtex"),
                "Abstract": ref.get("abstract"),
                "Methodological category": ref.get("methodological_category"),
                "Central technical idea": ref.get("central_technical_idea"),
                "Author-reported strengths": ref.get("author_reported_strengths"),
                "Reported limitations": ref.get("reported_limitations"),
                "Relevance to our work": ref.get("relevance_to_our_work"),
                "Assigned literature role": ref.get("introduction_paragraph_role"),
            }))
        commands.append(_step(
            "Write one Related Work paragraph integrating these references, whose assigned roles do not match the standard groups. Organize them by their supplied ideas and relevance. Cite each discussed work with a valid LaTeX \\cite{{key}} command taken from its BibTeX entry. Do not invent citation keys or claims. Return exactly one paragraph, with no heading.",
            [("Other related-work references", ref_fields)],
        ))
        titles.append("Other related work")

    if any(synthesis.get(key) for key in ("common_trends", "open_problems", "explicit_research_gap")):
        commands.append(_step(
            "Write the closing Related Work paragraph. Critically synthesize the literature trends and open problems, then state how they establish the research gap. Keep this discussion grounded in the supplied synthesis and do not repeat the paper's full contribution list. Return exactly one paragraph, with no heading.",
            [("Common trends", synthesis.get("common_trends")),
             ("Open problems", synthesis.get("open_problems")),
             ("Explicit research gap", synthesis.get("explicit_research_gap"))],
        ))
        titles.append("Critical synthesis and research gap")

    return commands, titles


def consultation_step(system_prompt, system_data, initial_context, input_list, titles, contexto_acumulado):
    for i, (prompt, data) in enumerate(input_list):
        input_step = (
            f"{initial_context}\n"
            f"--- OBJETIVE OF PARAGRAPH ---\n{prompt}\n\n"
            f"--- CONTENT DATA ---\n{data}\n\n"
            f"--- TEXT ALREADY WRITTEN (CONTINUE FROM HERE) ---\n\n{contexto_acumulado}"
        )
        
        print(f"Generating paragraph {i+1} of {len(input_list)}...")
        
        paragrafo=consult_with_deepchat(    system_data["base_url"],
                                            system_data["api_key"],
                                            system_data["model"],
                                            input_step,
                                            system_prompt)
        
        paragrafo = "\n%% "+ titles[i] + "\n"  + paragrafo + "\n"
        
        # Acumula o texto para servir de contexto para o próximo
        
        contexto_acumulado += paragrafo

    return contexto_acumulado

def consultation_in_depth2(system_prompt, system_data, data):
    
    initial_context = generate_initial_context(data)
    contexto_acumulado = ""


    input_list_1, titles_1 = generate_llm_introduction_commands(data) 
    contexto_acumulado+= "\n\\section{Introduction}\n"
    contexto_acumulado = consultation_step(system_prompt, system_data, initial_context, input_list_1, titles_1, contexto_acumulado)

    input_list_2, titles_2 = generate_llm_relatedworks_commands(data)
    contexto_acumulado+= "\n\\section{Related works}\n"
    contexto_acumulado = consultation_step(system_prompt, system_data, initial_context, input_list_2, titles_2, contexto_acumulado)

    return contexto_acumulado


def consultation_in_text2(system_prompt, data):

   
    msg  = "System PROMPT:\n" 
    msg += system_prompt + "\n"
    
    msg += generate_initial_context(data)
    
    ####################
    msg += "\n\n"
    msg += "\\section{Introduction}\n"
    
    input_list_1, titles_1 = generate_llm_introduction_commands(data)
    
    for i, (prompt, step_data) in enumerate(input_list_1):
        msg += "\n\n"
        msg += "%% " + titles_1[i] + "\n\n"
        msg += "```\n"
        msg += "PROMPT:\n"
        msg += prompt + "\n\n"
        msg += "DATA:\n"
        msg += step_data + "\n"
        msg += "```\n"
    
    ####################
    msg += "\n\n"
    msg += "\\section{Related works}\n"
    
    input_list_2, titles_2 = generate_llm_relatedworks_commands(data)

    for i, (prompt, step_data) in enumerate(input_list_2):
        msg += "\n\n"
        msg += "%% " + titles_2[i] + "\n\n"
        msg += "```\n"
        msg += "PROMPT:\n"
        msg += prompt + "\n\n"
        msg += "DATA:\n"
        msg += step_data + "\n"
        msg += "```\n"
    
    return msg
