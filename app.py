
import streamlit as st
import hashlib

from frontend.validation import validate_upload
from frontend.api import evaluate_resume


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ResumeIQ",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "submitted_files" not in st.session_state:
    st.session_state.submitted_files = set()


# =========================================================
# HEADER
# =========================================================

st.title("ResumeIQ")
st.caption("AI-powered Resume Evaluation System")

st.divider()


# =========================================================
# EVALUATION INPUTS
# =========================================================

st.subheader("Evaluate Your Resume")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=[
        "pdf",
        "docx",
        "txt",
        "doc",
        "rtf",
        "jpg",
        "jpeg",
        "png"
    ],
    help="Upload a supported resume file."
)

evaluation_type = st.radio(
    "Choose evaluation type",
    options=[
        "General Evaluation",
        "Role-Specific Evaluation"
    ],
    horizontal=True
)

role_name = None

if evaluation_type == "Role-Specific Evaluation":

    role_name = st.text_input(
        "Target Role",
        placeholder="Example: AI Engineer"
    )


# =========================================================
# EVALUATION
# =========================================================

if st.button(
    "Analyze Resume",
    type="primary",
    use_container_width=True
):

    # -----------------------------------------------------
    # 1. VALIDATE UPLOAD
    # -----------------------------------------------------

    is_valid, error_message = validate_upload(uploaded_file)

    if not is_valid:
        st.error(error_message)
        st.stop()

    # -----------------------------------------------------
    # 2. VALIDATE ROLE
    # -----------------------------------------------------

    if (
        evaluation_type == "Role-Specific Evaluation"
        and not role_name.strip()
    ):
        st.error("Please enter a target role.")
        st.stop()

    # -----------------------------------------------------
    # 3. READ FILE AND GENERATE HASH
    # -----------------------------------------------------

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    normalized_role = (
        role_name.strip()
        if evaluation_type == "Role-Specific Evaluation"
        else None
    )

    evaluation_mode = (
        "general"
        if evaluation_type == "General Evaluation"
        else "role_specific"
    )

    # -----------------------------------------------------
    # 4. CHECK DUPLICATE SUBMISSION
    # -----------------------------------------------------

    submission_key = (
        file_hash,
        evaluation_mode,
        normalized_role.lower() if normalized_role else None
    )

    if submission_key in st.session_state.submitted_files:

        st.warning(
            "This resume has already been evaluated "
            "using these evaluation settings in this session."
        )

        st.info(
            "Change the evaluation settings or upload a different resume."
        )

        st.stop()

    # -----------------------------------------------------
    # 5. REGISTER SUBMISSION
    # -----------------------------------------------------

    st.session_state.submitted_files.add(submission_key)

    # -----------------------------------------------------
    # 6. RUN EVALUATION THROUGH FASTAPI
    # -----------------------------------------------------

    try:

        with st.spinner("Analyzing your resume..."):

            result = evaluate_resume(
                uploaded_file=uploaded_file,
                evaluation_type=evaluation_mode,
                role_name=normalized_role
            )

        # -------------------------------------------------
        # 7. DISPLAY RESULTS
        # -------------------------------------------------

        st.success("Resume evaluation completed!")

        st.subheader("Evaluation Results")

        if isinstance(result, dict):
            result_data = result
        else:
            result_data = result

        st.json(result_data)

    except Exception as e:

        # Allow retry after a failed request.
        st.session_state.submitted_files.discard(submission_key)

        # Log technical details in the terminal.
        print(f"Resume evaluation failed: {e}")

        st.error(
            "Something went wrong while evaluating the resume. "
            "Please try again."
        )
