from deepeval.metrics.faithfulness.faithfulness import FaithfulnessTemplate
from deepeval.metrics.contextual_relevancy import ContextualRelevancyTemplate


class CustomFaithfulnessTemplate(FaithfulnessTemplate):

    @staticmethod
    def generate_claims(actual_output: str, **kwargs) -> str:

        return f"""
        Extract all distinct, atomic, factual claims about the candidate
        from the AI-generated resume evaluation.

        INCLUDE:
        - Education, certifications and qualifications.
        - Employment, job titles, responsibilities and experience.
        - Skills, technologies and proficiency levels.
        - Projects, implementation details, tools and methods.
        - Experiments, evaluation methods, data quality and outcomes.
        - Explicit claims about missing evidence or qualifications.

        EXCLUDE:
        - Scores, role labels and subjective suitability judgments.
        - Recommendations and future learning suggestions.
        - Opinions without independently verifiable factual content.

        RULES:
        1. Extract only claims explicitly stated in the output.
        2. Make each claim independently verifiable.
        3. Split compound claims into distinct factual assertions.
        4. Cover reasoning, strengths, skill gaps and all other sections.
        5. Preserve polarity, qualifiers and certainty.
        6. Distinguish listed skills from demonstrated proficiency.
        7. Do not infer implementation, achievements or expertise.
        8. Do not turn recommendations into claims about missing skills.
        9. Preserve contradictory claims instead of resolving them.
        10. Remove exact and semantic duplicates without losing distinct facts.
        11. Express missing evidence as a claim about the resume, not
            as proof that the candidate lacks the actual ability.

        Examples:
        - "Basic PyTorch knowledge" must remain basic.
        - "The resume lacks evidence of NLP" must remain a negative
          claim about resume evidence.
        - "Proposed an experiment" must not become "conducted an experiment".
        - "The resume lacks evidence of deep learning" and
          "The candidate has limited deep learning experience"
          must both be retained if both are stated.

        Return ONLY valid JSON:
        {{
            "claims": [
                "The candidate lists Python as a skill.",
                "The candidate implemented a RAG project.",
                "The resume provides no evidence of NLP experience."
            ]
        }}

        AI-generated output:
        {actual_output}

        JSON:
        """


    @staticmethod
    def generate_verdicts(
        claims: list[str],
        retrieval_context: list[str]
    ) -> str:

        return f"""
        Verify each claim independently using ONLY the supplied context.

        CONTEXT:
        {retrieval_context}

        SOURCE RULES:
        - Candidate Resume: evidence about the candidate.
        - Role Guide: job requirements, NOT evidence of candidate abilities.
        - Benchmark Resume: comparison material, NOT candidate evidence.
        - Never attribute benchmark or role-guide information to the candidate.

        VERDICTS:

        YES:
        - The claim is directly supported.
        - Its meaning and qualifiers match the evidence.
        - A claim about missing resume evidence is supported when the
          complete resume contains no relevant information.

        NO:
        - The claim is contradicted by the resume.
        - The resume contains evidence contradicting a missing-evidence claim.
        - The claim asserts information not supported by the supplied evidence.

        EVIDENCE RULES:
        1. A listed skill proves that the resume mentions it, not proficiency.
        2. Proficiency claims require evidence appropriate to their level.
        3. Projects support only their documented implementation details.
        4. Do not infer ownership, achievements, impact or expertise.
        5. Proposed work must not be treated as completed work.
        6. Absence from a resume does not prove actual absence of a skill.
        7. Match meaning rather than exact wording.
        8. Do not mark a compound claim YES if an important part is unsupported.
        9. When evidence is ambiguous or insufficient, use NO and explain why.

        CLAIMS:
        {claims}

        Return exactly one verdict per claim, in the original order.
        Do not omit or combine claims.

        Return ONLY valid JSON:
        {{
            "verdicts": [
                {{
                    "verdict": "yes",
                    "reason": "The resume explicitly documents this."
                }},
                {{
                    "verdict": "no",
                    "reason": "The resume does not support the claimed proficiency."
                }}
            ]
        }}

        JSON:
        """


    @staticmethod
    def generate_reason(
        score: float,
        contradictions: list,
        **kwargs
    ) -> str:

        return f"""
        Explain the Faithfulness score concisely.

        Score: {score}
        Failed claims: {contradictions}

        Distinguish unsupported claims from claims contradicted by evidence.
        Identify the main reasons for score reduction.
        Do not introduce additional issues.

        Return ONLY valid JSON:
        {{
            "reason": "..."
        }}
        """









class CustomAnswerRelevancyTemplate:

    @staticmethod
    def generate_statements(actual_output: str):

        return f"""
        Extract the distinct statements from this AI-generated
        resume evaluation.

        INCLUDE:
        - Candidate-related factual statements.
        - Final and individual criterion scores.
        - Evaluation conclusions and reasoning.
        - Strengths and skill gaps.
        - Recommendations and suggested improvements.

        RULES:
        1. Break compound statements into atomic statements.
        2. Preserve the original meaning.
        3. Do not add information or make inferences.
        4. Keep existing skills, missing skills and recommendations distinct.
        5. Do not omit statements simply because they are scores,
           conclusions or recommendations.
        6. Remove redundant duplicates.

        Return ONLY valid JSON:
        {{
            "statements": [
                "The candidate has experience with Python.",
                "The candidate received a technical skills score of 7.",
                "The candidate should improve their RAG knowledge."
            ]
        }}

        OUTPUT:
        {actual_output}

        JSON:
        """


    @staticmethod
    def generate_verdicts(input: str, statements: list):

        return f"""
        Evaluate whether each statement addresses the user's request.

        USER REQUEST:
        {input}

        STATEMENTS:
        {statements}

        YES:
        - Directly contributes to the requested resume evaluation.
        - Addresses scores, reasoning, strengths, skill gaps,
          qualifications or recommendations.

        NO:
        - Is unrelated to the requested evaluation.

        RULES:
        1. Evaluate relevance, NOT factual accuracy.
        2. Judge every statement independently.
        3. Do not reject relevant recommendations or scores.
        4. Do not confuse recommendations with existing candidate skills.
        5. Return exactly one verdict per statement, preserving order.

        Return ONLY valid JSON:
        {{
            "verdicts": [
                {{"verdict": "yes"}},
                {{"verdict": "no", "reason": "Unrelated to the resume evaluation."}}
            ]
        }}

        JSON:
        """





class CustomContextualRelevancyTemplate(ContextualRelevancyTemplate):

    @staticmethod
    def generate_verdicts(input: str, context: str) -> str:

        return f"""
        Evaluate whether each piece of context is useful for assessing
        a candidate's suitability for the requested role.

        TASK:
        {input}

        CONTEXT:
        {context}

        IDENTIFY EACH SOURCE:
        - Candidate Resume
        - Role Guide
        - Retrieved Benchmark Resume

        SOURCE RULES:

        Candidate Resume:
        - Skills, education, projects, experience and achievements
          are relevant professional evidence.
        - Personal contact details and unrelated personal information
          are irrelevant.

        Role Guide:
        - Required skills, responsibilities, qualifications and
          professional expectations are relevant.
        - These are role requirements, NOT candidate evidence.

        Benchmark Resume:
        - Skills, projects, education and experience are relevant
          comparison information.
        - Never attribute benchmark information to the candidate.

        GENERAL RULES:
        1. Relevance is not the same as candidate quality or factual accuracy.
        2. Weak candidate evidence can still be relevant.
        3. Transferable professional skills can be relevant.
        4. Ignore null, empty and missing values.
        5. Do not invent information.
        6. Prefix statements with their source.
        7. Convert structured information into concise readable statements.
        8. Evaluate each meaningful information item independently.

        YES: Useful professional evidence, role requirements or comparison data.
        NO: Unrelated information or irrelevant personal details.

        Return ONLY valid JSON:
        {{
            "verdicts": [
                {{
                    "statement": "Candidate Resume — Python is listed as a skill.",
                    "verdict": "yes",
                    "reason": ""
                }},
                {{
                    "statement": "Candidate Resume — Personal email address.",
                    "verdict": "no",
                    "reason": "Unrelated personal contact information."
                }}
            ]
        }}

        JSON:
        """


    @staticmethod
    def generate_reason(
        input: str,
        irrelevant_statements: list,
        relevant_statements: list,
        score: str,
        multimodal: bool
    ) -> str:

        return f"""
        Explain the contextual relevancy score.

        Task: {input}
        Score: {score}
        Relevant: {relevant_statements}
        Irrelevant: {irrelevant_statements}

        Explain why the context is useful or irrelevant.
        Preserve the distinction between candidate evidence,
        role requirements and benchmark comparison material.
        Never attribute benchmark or role-guide information to the candidate.
        Do not evaluate factual accuracy or candidate quality.

        Return a concise explanation.
        """