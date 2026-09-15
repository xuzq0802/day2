import streamlit as st 
import pandas as pd
from io import StringIO
from pypdf import PdfReader
from pathlib import Path

st.title("Exercise 2.1")

#1.Allows the user to upload a document

uploaded_file = st.file_uploader("Choose a file",type=["pdf", "txt"],)
if  uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        #st.write(text)
    else:
        bytes_data = uploaded_file.getvalue()
        string_data = bytes_data.decode("utf-8")
        st.write(string_data)

#want a dropdown to allow me to select previously chunked flles andlook at them. For example, if Irestart my appllication, I don't want to have to re-read and re-chunkeverything if I've already done it)
# Browse previously saved chunks.
st.divider()
st.subheader("Previously saved documents")

# This Python file is inside the pages folder.
project_dir = Path(__file__).resolve().parent.parent

saved_folders = sorted(
    [
        folder
        for folder in project_dir.glob("Chunks - *")
        if folder.is_dir()
    ],
    key=lambda folder: folder.name,
)

if not saved_folders:
    st.info("No saved documents found. Upload and save a document first.")
else:
    selected_folder = st.selectbox(
        "Choose a saved document",
        options=saved_folders,
        format_func=lambda folder: folder.name,
    )

    chunk_files = sorted(
        selected_folder.glob("chunk_*.txt"),
        key=lambda file: (
            int(file.stem.split("_")[-1])
            if file.stem.split("_")[-1].isdigit()
            else float("inf")
        ),
    )

    st.write(f"{len(chunk_files)} saved chunks")

    for chunk_file in chunk_files:
        with st.expander(chunk_file.name):
            try:
                content = chunk_file.read_text(encoding="utf-8")
                st.text(content)
            except (OSError, UnicodeError) as error:
                st.error(f"Unable to read this chunk: {error}")

# 2. Allows the user to chunk the document.
if uploaded_file is not None:
    import re

    is_pdf = uploaded_file.type == "application/pdf"
    document_text = text if is_pdf else string_data

    if not document_text.strip():
        st.warning("No readable text was found in this document.")
        st.stop()

    method = st.selectbox(
        "Choose a chunking method",
        ["Paragraph", "Page", "Heading", "Character count"],
    )

    if method == "Paragraph":
        st.caption(
            "Splits at blank lines. PDF paragraph boundaries "
            "may not be preserved during text extraction."
        )

        chunks = [
            part.strip()
            for part in re.split(r"\n\s*\n", document_text)
            if part.strip()
        ]

    elif method == "Page":
        if is_pdf:
            pages = [
                page.extract_text() or ""
                for page in reader.pages
            ]
        else:
            pages = document_text.split("\f")
            st.caption(
                "Text files are split at form-feed characters. "
                "Without them, the file is treated as one page."
            )

        chunks = [
            page.strip()
            for page in pages
            if page.strip()
        ]

    elif method == "Heading":
        heading_level = st.selectbox(
            "Split at heading levels",
            ["Level 1 only", "Levels 1 and 2"],
        )

        max_level = 1 if heading_level == "Level 1 only" else 2

        st.caption(
            "Detects Markdown headings and numbered headings "
            "such as '1 Introduction' and '1.1 Background'. "
            "Font size and bold formatting are not detected. "
            "Numbered lists may be mistaken for headings."
        )

        chunks = []
        current_lines = []

        for line in document_text.splitlines():
            stripped = line.strip()
            level = None

            markdown_match = re.match(
                r"^(#{1,6})\s+\S", stripped
            )
            numbered_match = re.match(
                r"^(\d+(?:\.\d+)*)(?:[.)])?\s+\S",
                stripped,
            )

            if markdown_match:
                level = len(markdown_match.group(1))
            elif numbered_match:
                level = len(
                    numbered_match.group(1).split(".")
                )

            if level is not None and level <= max_level:
                previous_chunk = "\n".join(current_lines).strip()

                if previous_chunk:
                    chunks.append(previous_chunk)

                current_lines = [line]
            else:
                current_lines.append(line)

        last_chunk = "\n".join(current_lines).strip()

        if last_chunk:
            chunks.append(last_chunk)

        if len(chunks) == 1:
            st.info("No additional heading boundaries were detected.")

    else:
        chunk_size = int(
            st.number_input(
                "Maximum characters per chunk",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
            )
        )

        st.caption("This method may split sentences between chunks.")

        chunks = [
            document_text[i:i + chunk_size]
            for i in range(0, len(document_text), chunk_size)
            if document_text[i:i + chunk_size].strip()
        ]

    # Display each chunk in a collapsible section.
    st.success(f"Created {len(chunks)} chunks.")

    for index, chunk in enumerate(chunks, start=1):
        with st.expander(
            f"Chunk {index} - {len(chunk)} characters"
        ):
            st.text(chunk)

    # 3. Saves each chunk of the document to the project directory.
        # 3. Save chunks in a folder named after the document.
    if st.button("Save all chunks"):
        import re
        from datetime import datetime
        from uuid import uuid4

        # This Python file is inside the pages folder.
        project_dir = Path(__file__).resolve().parent.parent

        document_name = Path(uploaded_file.name).stem

        # Remove characters that are invalid in folder names.
        safe_name = re.sub(
            r'[<>:"/\\|?*\x00-\x1f]', "_", document_name
        ).strip(" .")[:100] or "Document"

        # Use a unique folder for each save.
        save_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid4().hex[:6]
        )

        output_dir = (
            project_dir / f"Chunks - {safe_name} - {save_id}"
        )

        try:
            output_dir.mkdir(parents=True, exist_ok=False)

            for index, chunk in enumerate(chunks, start=1):
                file_path = output_dir / f"chunk_{index:03d}.txt"
                file_path.write_text(chunk, encoding="utf-8")

        except OSError as error:
            st.error(f"Unable to save all chunks: {error}")
        else:
            st.success(f"Saved chunks to: {output_dir}")

#4.Reads the first chunk you have saved and stores it in a variable.
#5.Displays the first chunk back to the user