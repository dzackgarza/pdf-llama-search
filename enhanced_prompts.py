#!/usr/bin/env python3
"""
Enhanced prompt templates for mathematical research paper queries.
Specialized prompts for theorems, definitions, lemmas, proofs, and mathematical concepts.
"""

from typing import List, Dict
from haystack.dataclasses import Document


def detect_query_type(question: str) -> str:
    """Detect the type of mathematical query to use the appropriate prompt template."""
    question_lower = question.lower()
    
    # Theorem/result queries
    if any(term in question_lower for term in ['theorem', 'theorems', 'result', 'results', 'proposition', 'propositions']):
        return "theorem"
    
    # Definition queries
    if any(term in question_lower for term in ['definition', 'definitions', 'define', 'what is', 'meaning of']):
        return "definition"
    
    # Lemma queries
    if any(term in question_lower for term in ['lemma', 'lemmas', 'preliminary result']):
        return "lemma"
    
    # Proof queries
    if any(term in question_lower for term in ['proof', 'proofs', 'prove', 'demonstration', 'show that']):
        return "proof"
    
    # Mathematical concept queries
    if any(term in question_lower for term in ['concept', 'concepts', 'notion', 'theory', 'approach', 'method']):
        return "concept"
    
    # Application/example queries
    if any(term in question_lower for term in ['example', 'examples', 'application', 'applications', 'use case']):
        return "application"
    
    # Default to general mathematical
    return "general"


def create_theorem_prompt(question: str, documents: List[Document]) -> str:
    """Create a prompt specifically for theorem and result queries."""
    
    prompt_parts = [
        "You are a mathematical research expert specializing in analyzing academic papers. ",
        "Provide comprehensive information about mathematical theorems and results with precise citations.\n\n",
        
        "## Response Structure for Theorems:\n",
        "1. **Theorem Statement**: Complete and precise statement as given in the paper\n",
        "2. **Context & Significance**: Where this theorem fits in the broader mathematical landscape\n",
        "3. **Assumptions & Conditions**: All hypotheses and required conditions\n",
        "4. **Implications & Consequences**: What this theorem implies or enables\n",
        "5. **Related Results**: Connected theorems, corollaries, or generalizations\n",
        "6. **Historical Context**: Attribution and development history if mentioned\n\n",
        
        "## Requirements:\n",
        "- **PRECISE CITATIONS**: Include exact page numbers, section numbers, theorem numbers\n",
        "- **MATHEMATICAL NOTATION**: Use proper LaTeX formatting:\n",
        "  * Inline math: $x^2$, $\\alpha \\in \\mathbb{R}$, $f: X \\to Y$\n",
        "  * Display math: $$\\int_0^1 f(x) dx = \\sum_{i=1}^n a_i$$\n",
        "  * Use standard symbols: \\mathbb{R}, \\mathbb{C}, \\mathbb{Z}, \\mathbb{Q}, \\mathbb{N}\n",
        "- **EXACT REFERENCES**: Format as [Paper: filename, Page X, Section Y.Z, Theorem N]\n",
        "- **COMPLETE STATEMENTS**: Include all conditions and conclusions\n",
        "- **MATHEMATICAL RIGOR**: Maintain the precision of the original statements\n\n",
        
        "## Source Documents:\n"
    ]
    
    # Add document context with page information
    for i, doc in enumerate(documents, 1):
        filename = doc.meta.get("name", "Unknown")
        page = doc.meta.get("page", "?")
        content = doc.content[:800] + "..." if len(doc.content) > 800 else doc.content
        
        prompt_parts.append(f"**Document {i}:** {filename} (Page {page})\n```\n{content}\n```\n\n")
    
    prompt_parts.extend([
        f"## Question: {question}\n\n",
        "## Mathematical Analysis:\n",
        "Provide a comprehensive analysis following the structure above. Focus on:\n",
        "- Exact theorem statements with proper mathematical notation\n",
        "- Precise page and section references for each result\n",
        "- Mathematical significance and relationships to other results\n",
        "- All conditions, assumptions, and scope of applicability\n\n"
    ])
    
    return "".join(prompt_parts)


def create_definition_prompt(question: str, documents: List[Document]) -> str:
    """Create a prompt specifically for mathematical definition queries."""
    
    prompt_parts = [
        "You are a mathematical research expert focusing on precise definitions and terminology. ",
        "Provide comprehensive information about mathematical definitions with exact citations.\n\n",
        
        "## Response Structure for Definitions:\n",
        "1. **Formal Definition**: Exact definition as stated in the paper\n",
        "2. **Intuitive Explanation**: Clear, accessible explanation of the concept\n",
        "3. **Mathematical Properties**: Key properties and characteristics\n",
        "4. **Examples & Non-examples**: Illustrative examples and counterexamples\n",
        "5. **Related Concepts**: Connected definitions and terminology\n",
        "6. **Usage Context**: How this definition is used in the paper\n\n",
        
        "## Requirements:\n",
        "- **VERBATIM DEFINITIONS**: Quote definitions exactly as they appear\n",
        "- **PRECISE NOTATION**: Preserve all mathematical symbols and formatting\n",
        "- **COMPLETE CONTEXT**: Include all qualifying conditions and scope\n",
        "- **CLEAR CITATIONS**: [Paper: filename, Page X, Definition Y] format\n",
        "- **MATHEMATICAL SYMBOLS**: Proper LaTeX: $\\mathbb{R}$, $\\in$, $\\subset$, $\\forall$, $\\exists$\n\n",
        
        "## Source Documents:\n"
    ]
    
    # Add document context
    for i, doc in enumerate(documents, 1):
        filename = doc.meta.get("name", "Unknown")
        page = doc.meta.get("page", "?")
        content = doc.content[:700] + "..." if len(doc.content) > 700 else doc.content
        
        prompt_parts.append(f"**Document {i}:** {filename} (Page {page})\n```\n{content}\n```\n\n")
    
    prompt_parts.extend([
        f"## Question: {question}\n\n",
        "## Definition Analysis:\n",
        "Provide a complete analysis of the mathematical definition(s) with:\n",
        "- Exact quoted definitions with proper mathematical notation\n",
        "- Clear explanations accessible to researchers in the field\n",
        "- Precise citations with page numbers and definition numbers\n",
        "- Context of how these definitions are used in the research\n\n"
    ])
    
    return "".join(prompt_parts)


def create_proof_prompt(question: str, documents: List[Document]) -> str:
    """Create a prompt specifically for proof-related queries."""
    
    prompt_parts = [
        "You are a mathematical research expert specializing in proof analysis. ",
        "Provide detailed information about mathematical proofs with precise structure and citations.\n\n",
        
        "## Response Structure for Proofs:\n",
        "1. **Proof Overview**: High-level strategy and approach\n",
        "2. **Key Steps**: Main logical steps in the proof\n",
        "3. **Techniques Used**: Mathematical methods and tools employed\n",
        "4. **Critical Insights**: Key ideas that make the proof work\n",
        "5. **Dependencies**: Required lemmas, theorems, or definitions\n",
        "6. **Proof Structure**: Outline of the logical flow\n\n",
        
        "## Requirements:\n",
        "- **PROOF STRUCTURE**: Clearly outline the logical progression\n",
        "- **MATHEMATICAL RIGOR**: Maintain the precision of the original proof\n",
        "- **KEY TECHNIQUES**: Identify and explain proof methods used\n",
        "- **EXACT CITATIONS**: [Paper: filename, Page X, Proof of Theorem Y]\n",
        "- **MATHEMATICAL NOTATION**: Proper LaTeX formatting throughout\n\n",
        
        "## Source Documents:\n"
    ]
    
    # Add document context
    for i, doc in enumerate(documents, 1):
        filename = doc.meta.get("name", "Unknown")
        page = doc.meta.get("page", "?")
        content = doc.content[:600] + "..." if len(doc.content) > 600 else doc.content
        
        prompt_parts.append(f"**Document {i}:** {filename} (Page {page})\n```\n{content}\n```\n\n")
    
    prompt_parts.extend([
        f"## Question: {question}\n\n",
        "## Proof Analysis:\n",
        "Provide a comprehensive analysis of the proof(s) with:\n",
        "- Clear outline of the proof strategy and main steps\n",
        "- Identification of key mathematical techniques and insights\n",
        "- Precise citations to specific pages and proof sections\n",
        "- Mathematical notation preserved from the original\n\n"
    ])
    
    return "".join(prompt_parts)


def create_concept_prompt(question: str, documents: List[Document]) -> str:
    """Create a prompt for mathematical concept and theory queries."""
    
    prompt_parts = [
        "You are a mathematical research expert focusing on mathematical concepts and theories. ",
        "Provide comprehensive explanations of mathematical ideas with proper academic context.\n\n",
        
        "## Response Structure for Concepts:\n",
        "1. **Concept Introduction**: Clear explanation of the mathematical concept\n",
        "2. **Formal Framework**: Mathematical structure and formalization\n",
        "3. **Key Properties**: Important characteristics and behavior\n",
        "4. **Applications**: How the concept is used in mathematics\n",
        "5. **Development**: Historical context and evolution of the idea\n",
        "6. **Related Areas**: Connections to other mathematical fields\n\n",
        
        "## Source Documents:\n"
    ]
    
    # Add document context
    for i, doc in enumerate(documents, 1):
        filename = doc.meta.get("name", "Unknown")
        page = doc.meta.get("page", "?")
        content = doc.content[:600] + "..." if len(doc.content) > 600 else doc.content
        
        prompt_parts.append(f"**Document {i}:** {filename} (Page {page})\n```\n{content}\n```\n\n")
    
    prompt_parts.extend([
        f"## Question: {question}\n\n",
        "## Concept Analysis:\n",
        "Provide a thorough explanation of the mathematical concept with:\n",
        "- Clear, accessible explanations with proper mathematical notation\n",
        "- Formal mathematical structure using LaTeX: $\\mathbb{R}$, $\\subset$, $\\mapsto$\n",
        "- Precise citations with page numbers and section references\n",
        "- Context of how the concept fits into the broader mathematical landscape\n\n"
    ])
    
    return "".join(prompt_parts)


def create_enhanced_prompt_adaptive(question: str, documents: List[Document]) -> str:
    """Create an adaptive prompt based on the type of mathematical query."""
    
    query_type = detect_query_type(question)
    
    if query_type == "theorem":
        return create_theorem_prompt(question, documents)
    elif query_type == "definition":
        return create_definition_prompt(question, documents)
    elif query_type == "proof":
        return create_proof_prompt(question, documents)
    elif query_type == "concept":
        return create_concept_prompt(question, documents)
    else:
        # Use a general mathematical prompt
        return create_general_mathematical_prompt(question, documents)


def create_general_mathematical_prompt(question: str, documents: List[Document]) -> str:
    """Create a general prompt for mathematical research queries."""
    
    prompt_parts = [
        "You are a mathematical research expert analyzing academic papers. ",
        "Provide comprehensive, well-cited responses to mathematical research questions.\n\n",
        
        "## Response Requirements:\n",
        "- **PRECISE CITATIONS**: Include exact page numbers, section numbers, theorem/definition numbers\n",
        "- **MATHEMATICAL NOTATION**: Use proper LaTeX formatting for all mathematical expressions\n",
        "- **COMPREHENSIVE COVERAGE**: Address all aspects of the question with supporting evidence\n",
        "- **RESEARCH CONTEXT**: Explain the significance within the mathematical field\n",
        "- **EXACT QUOTES**: When referencing definitions or theorems, quote them precisely\n\n",
        
        "## Citation Format:\n",
        "Use: [Paper: filename, Page X, Section Y.Z] or [Paper: filename, Page X, Theorem/Definition N]\n\n",
        
        "## Source Documents:\n"
    ]
    
    # Add document context
    for i, doc in enumerate(documents, 1):
        filename = doc.meta.get("name", "Unknown")
        page = doc.meta.get("page", "?")
        content = doc.content[:700] + "..." if len(doc.content) > 700 else doc.content
        
        prompt_parts.append(f"**Document {i}:** {filename} (Page {page})\n```\n{content}\n```\n\n")
    
    prompt_parts.extend([
        f"## Question: {question}\n\n",
        "## Mathematical Research Response:\n",
        "Provide a comprehensive response that:\n",
        "- Directly addresses the question with mathematical precision\n",
        "- Uses proper LaTeX notation: $\\mathbb{R}$, $\\in$, $\\forall$, $\\exists$, etc.\n",
        "- Includes exact citations with page numbers and section references\n",
        "- Maintains academic rigor and mathematical accuracy\n",
        "- Explains the research significance and context\n\n"
    ])
    
    return "".join(prompt_parts)


def get_prompt_analysis(question: str) -> Dict[str, str]:
    """Analyze the question and return metadata about the prompt strategy."""
    query_type = detect_query_type(question)
    
    analysis = {
        "query_type": query_type,
        "focus_areas": get_focus_areas(query_type),
        "citation_requirements": "Exact page numbers, section numbers, theorem/definition numbers",
        "mathematical_notation": "LaTeX formatting required",
        "response_structure": get_response_structure(query_type)
    }
    
    return analysis


def get_focus_areas(query_type: str) -> List[str]:
    """Get the focus areas for different query types."""
    focus_map = {
        "theorem": ["exact statement", "conditions", "implications", "proof techniques"],
        "definition": ["formal definition", "examples", "properties", "usage context"],
        "proof": ["proof strategy", "key steps", "techniques", "logical structure"],
        "concept": ["mathematical framework", "properties", "applications", "connections"],
        "lemma": ["statement", "role in larger proof", "conditions", "usage"],
        "application": ["concrete examples", "implementation", "practical usage"],
        "general": ["mathematical precision", "research context", "exact citations"]
    }
    
    return focus_map.get(query_type, focus_map["general"])


def get_response_structure(query_type: str) -> str:
    """Get the recommended response structure for different query types."""
    structure_map = {
        "theorem": "Statement → Context → Conditions → Implications → Related Results",
        "definition": "Formal Definition → Explanation → Properties → Examples → Context",
        "proof": "Overview → Key Steps → Techniques → Insights → Dependencies",
        "concept": "Introduction → Framework → Properties → Applications → Connections",
        "general": "Direct Response → Mathematical Details → Citations → Context"
    }
    
    return structure_map.get(query_type, structure_map["general"]) 