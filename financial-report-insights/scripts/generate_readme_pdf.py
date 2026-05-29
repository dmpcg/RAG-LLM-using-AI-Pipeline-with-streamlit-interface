#!/usr/bin/env python3
"""
PDF Generator for RAG-LLM Project README
Creates a professional, aesthetically pleasing PDF from the README content.

Design System: "Technical Clarity"
- Deep Blue (#1E40AF) for headers
- Teal (#0D9488) for highlights
- Dark gray (#1F2937) for code blocks
- Near-black (#111827) for body text
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# Color Palette - Technical Clarity Design System
DEEP_BLUE = colors.HexColor("#1E40AF")
TEAL = colors.HexColor("#0D9488")
CODE_BG = colors.HexColor("#1F2937")
BODY_TEXT = colors.HexColor("#111827")
LIGHT_GRAY = colors.HexColor("#F3F4F6")
CODE_TEXT = colors.HexColor("#E5E7EB")
ACCENT_LIGHT = colors.HexColor("#DBEAFE")

def create_styles():
    """Create custom paragraph styles for the document."""
    styles = getSampleStyleSheet()

    # Main Title Style
    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=DEEP_BLUE,
        spaceAfter=6,
        spaceBefore=0,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold',
        leading=34
    ))

    # Subtitle Style
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=20,
        alignment=TA_LEFT,
        fontName='Helvetica',
        leading=16
    ))

    # Section Header Style
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=DEEP_BLUE,
        spaceBefore=24,
        spaceAfter=12,
        fontName='Helvetica-Bold',
        borderPadding=4,
        leading=20
    ))

    # Body Text Style
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=BODY_TEXT,
        spaceBefore=6,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        leading=14
    ))

    # Feature Item Style
    styles.add(ParagraphStyle(
        name='FeatureItem',
        parent=styles['Normal'],
        fontSize=10,
        textColor=BODY_TEXT,
        spaceBefore=4,
        spaceAfter=4,
        leftIndent=20,
        fontName='Helvetica',
        leading=14
    ))

    # Code Block Style
    styles.add(ParagraphStyle(
        name='CodeBlock',
        parent=styles['Normal'],
        fontSize=9,
        textColor=CODE_TEXT,
        backColor=CODE_BG,
        spaceBefore=8,
        spaceAfter=8,
        leftIndent=12,
        rightIndent=12,
        fontName='Courier',
        leading=13,
        borderPadding=10
    ))

    # Step Number Style
    styles.add(ParagraphStyle(
        name='StepNumber',
        parent=styles['Normal'],
        fontSize=11,
        textColor=TEAL,
        fontName='Helvetica-Bold',
        spaceBefore=12,
        spaceAfter=4,
        leading=14
    ))

    # Technology Badge Style
    styles.add(ParagraphStyle(
        name='TechBadge',
        parent=styles['Normal'],
        fontSize=10,
        textColor=DEEP_BLUE,
        fontName='Helvetica-Bold',
        leading=12
    ))

    # Footer Style
    styles.add(ParagraphStyle(
        name='Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor("#6B7280"),
        alignment=TA_CENTER,
        fontName='Helvetica',
        leading=12
    ))

    # TOC Style
    styles.add(ParagraphStyle(
        name='TOCEntry',
        parent=styles['Normal'],
        fontSize=10,
        textColor=BODY_TEXT,
        spaceBefore=4,
        spaceAfter=4,
        leftIndent=15,
        fontName='Helvetica',
        leading=14
    ))

    return styles

def create_code_block(code_text, styles):
    """Create a styled code block with dark background."""
    # Create a table to simulate rounded corners effect
    code_para = Paragraph(
        f'<font face="Courier" color="#E5E7EB">{code_text}</font>',
        styles['CodeBlock']
    )

    code_table = Table([[code_para]], colWidths=[6.3*inch])
    code_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CODE_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    return code_table

def create_feature_box(features, styles):
    """Create a styled feature list with checkmarks."""
    elements = []

    for feature in features:
        # Split feature into title and description
        if "**" in feature:
            parts = feature.split("**")
            if len(parts) >= 3:
                title = parts[1]
                desc = parts[2].lstrip(": ")
                feature_text = f'<font color="#0D9488"><b>&#x2713;</b></font>  <b>{title}:</b> {desc}'
            else:
                feature_text = f'<font color="#0D9488"><b>&#x2713;</b></font>  {feature}'
        else:
            feature_text = f'<font color="#0D9488"><b>&#x2713;</b></font>  {feature}'

        elements.append(Paragraph(feature_text, styles['FeatureItem']))

    return elements

def create_tech_table(technologies, styles):
    """Create a styled technology grid."""
    tech_data = []
    row = []

    for i, tech in enumerate(technologies):
        if "**" in tech:
            parts = tech.split("**")
            if len(parts) >= 3:
                name = parts[1]
                desc = parts[2].lstrip(": ")
                cell_content = Paragraph(
                    f'<b><font color="#1E40AF">{name}</font></b><br/>'
                    f'<font size="9" color="#4B5563">{desc}</font>',
                    styles['CustomBody']
                )
            else:
                cell_content = Paragraph(tech, styles['CustomBody'])
        else:
            cell_content = Paragraph(tech, styles['CustomBody'])

        row.append(cell_content)

        if len(row) == 2 or i == len(technologies) - 1:
            if len(row) == 1:
                row.append("")
            tech_data.append(row)
            row = []

    if tech_data:
        tech_table = Table(tech_data, colWidths=[3.15*inch, 3.15*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ]))
        return tech_table

    return None

def build_pdf(output_path):
    """Build the complete PDF document."""

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = create_styles()
    story = []

    # ============================================
    # COVER / HEADER SECTION
    # ============================================

    # Title with emoji
    story.append(Paragraph(
        "RAG-LLM Using AI Pipeline with<br/>Streamlit Interface",
        styles['MainTitle']
    ))

    # Subtitle/tagline
    story.append(Paragraph(
        "Retrieval-Augmented Generation powered by Claude Sonet 3.5 LLM and Pathway Framework",
        styles['Subtitle']
    ))

    # Quick overview box
    overview_text = """
    <b>What it does:</b> Integrates RAG with Claude Sonet 3.5 LLM and Pathway framework to provide
    real-time insights into financial reports and tables. Ingests data from Google Drive and
    presents results through a user-friendly Streamlit interface.
    """

    overview_para = Paragraph(overview_text.strip(), styles['CustomBody'])
    overview_table = Table([[overview_para]], colWidths=[6.3*inch])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), ACCENT_LIGHT),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 20))

    # Technology badges row
    tech_badges = "Python  |  Streamlit  |  Claude LLM  |  Pathway  |  Google Drive API  |  Vector DB"
    story.append(Paragraph(
        f'<font color="#0D9488"><b>{tech_badges}</b></font>',
        ParagraphStyle('TechRow', parent=styles['CustomBody'], alignment=TA_CENTER, fontSize=10)
    ))

    story.append(Spacer(1, 10))

    # ============================================
    # TABLE OF CONTENTS
    # ============================================

    story.append(Paragraph("Table of Contents", styles['SectionHeader']))

    toc_items = [
        "1. Introduction",
        "2. Features",
        "3. Technologies Used",
        "4. Installation",
        "5. Usage",
        "6. Contributing",
        "7. License",
        "8. Contact"
    ]

    for item in toc_items:
        story.append(Paragraph(
            f'<font color="#1E40AF">{item}</font>',
            styles['TOCEntry']
        ))

    story.append(Spacer(1, 10))

    # ============================================
    # INTRODUCTION
    # ============================================

    story.append(Paragraph("1. Introduction", styles['SectionHeader']))

    intro_text = """
    In the world of finance, analyzing data efficiently is crucial. This project aims to streamline
    that process by combining advanced AI techniques with practical tools. By leveraging the power
    of <b>Retrieval-Augmented Generation (RAG)</b> and the <b>Claude Sonet 3.5 LLM</b>, users can
    extract meaningful insights from complex financial documents. The integration with
    <b>Google Drive</b> allows for easy data access, while the <b>Streamlit</b> interface makes
    it simple to visualize results.
    """
    story.append(Paragraph(intro_text.strip(), styles['CustomBody']))

    # ============================================
    # FEATURES
    # ============================================

    story.append(Paragraph("2. Features", styles['SectionHeader']))

    features = [
        "**Real-time Data Processing**: Ingest and analyze data from Google Drive instantly.",
        "**Structured and Unstructured Format Handling**: Process various types of financial documents.",
        "**User-friendly Interface**: Visualize insights through a Streamlit-powered dashboard.",
        "**Integration with Claude Sonet 3.5 LLM**: Utilize advanced language models for enhanced analysis.",
        "**Retrieval-Augmented Generation**: Combine traditional data retrieval with modern AI techniques for better results."
    ]

    story.extend(create_feature_box(features, styles))

    # ============================================
    # TECHNOLOGIES USED
    # ============================================

    story.append(Paragraph("3. Technologies Used", styles['SectionHeader']))

    story.append(Paragraph(
        "This project employs a variety of technologies to achieve its goals:",
        styles['CustomBody']
    ))
    story.append(Spacer(1, 8))

    technologies = [
        "**Python**: The primary programming language for development.",
        "**Streamlit**: For creating the web interface.",
        "**Claude Sonet 3.5 LLM**: The language model for processing and generating text.",
        "**Pathway Framework**: To streamline the AI pipeline.",
        "**Google Drive API**: For data ingestion.",
        "**Vector Database**: For efficient data storage and retrieval."
    ]

    tech_table = create_tech_table(technologies, styles)
    if tech_table:
        story.append(tech_table)

    # ============================================
    # INSTALLATION
    # ============================================

    story.append(Paragraph("4. Installation", styles['SectionHeader']))

    story.append(Paragraph(
        "To set up the project locally, follow these steps:",
        styles['CustomBody']
    ))

    # Step 1
    story.append(Paragraph("Step 1: Clone the Repository", styles['StepNumber']))
    story.append(create_code_block(
        "git clone https://github.com/Dono1901/RAG-LLM-using-AI-Pipeline-with-streamlit-interface.git",
        styles
    ))

    # Step 2
    story.append(Paragraph("Step 2: Navigate to the Project Directory", styles['StepNumber']))
    story.append(create_code_block(
        "cd RAG-LLM-using-AI-Pipeline-with-streamlit-interface",
        styles
    ))

    # Step 3
    story.append(Paragraph("Step 3: Install Dependencies", styles['StepNumber']))
    story.append(Paragraph(
        "Make sure you have Python 3.8 or higher installed. Then, run:",
        styles['CustomBody']
    ))
    story.append(create_code_block("pip install -r requirements.txt", styles))

    # Step 4
    story.append(Paragraph("Step 4: Set Up Google Drive API", styles['StepNumber']))
    story.append(Paragraph(
        "Follow the instructions in the Google Drive API documentation to set up your credentials:<br/>"
        "<font color='#1E40AF'>https://developers.google.com/drive/api/v3/quickstart/python</font>",
        styles['CustomBody']
    ))

    # Step 5
    story.append(Paragraph("Step 5: Run the Application", styles['StepNumber']))
    story.append(Paragraph("Start the Streamlit server with:", styles['CustomBody']))
    story.append(create_code_block("streamlit run app.py", styles))

    # ============================================
    # USAGE
    # ============================================

    story.append(Paragraph("5. Usage", styles['SectionHeader']))

    story.append(Paragraph(
        "After setting up the application, you can start using it:",
        styles['CustomBody']
    ))

    usage_steps = [
        ("<b>Access the Interface:</b> Open your web browser and go to "
         "<font color='#1E40AF'>http://localhost:8501</font>"),
        "<b>Upload Financial Reports:</b> Use the interface to upload your financial documents from Google Drive.",
        "<b>Analyze Data:</b> The system will process the data and provide insights in real time.",
        "<b>Visualize Results:</b> Explore the insights through the interactive dashboard."
    ]

    for i, step in enumerate(usage_steps, 1):
        story.append(Paragraph(
            f'<font color="#0D9488"><b>{i}.</b></font>  {step}',
            styles['FeatureItem']
        ))

    # ============================================
    # CONTRIBUTING
    # ============================================

    story.append(Paragraph("6. Contributing", styles['SectionHeader']))

    story.append(Paragraph(
        "We welcome contributions! If you want to help improve this project, please follow these steps:",
        styles['CustomBody']
    ))

    story.append(Paragraph(
        '<font color="#0D9488"><b>1.</b></font>  <b>Fork the Repository</b>',
        styles['FeatureItem']
    ))

    story.append(Paragraph(
        '<font color="#0D9488"><b>2.</b></font>  <b>Create a New Branch:</b>',
        styles['FeatureItem']
    ))
    story.append(create_code_block("git checkout -b feature/YourFeature", styles))

    story.append(Paragraph(
        '<font color="#0D9488"><b>3.</b></font>  <b>Make Your Changes</b>',
        styles['FeatureItem']
    ))

    story.append(Paragraph(
        '<font color="#0D9488"><b>4.</b></font>  <b>Commit Your Changes:</b>',
        styles['FeatureItem']
    ))
    story.append(create_code_block('git commit -m "Add some feature"', styles))

    story.append(Paragraph(
        '<font color="#0D9488"><b>5.</b></font>  <b>Push to the Branch:</b>',
        styles['FeatureItem']
    ))
    story.append(create_code_block("git push origin feature/YourFeature", styles))

    story.append(Paragraph(
        '<font color="#0D9488"><b>6.</b></font>  <b>Open a Pull Request</b>',
        styles['FeatureItem']
    ))

    # ============================================
    # LICENSE
    # ============================================

    story.append(Paragraph("7. License", styles['SectionHeader']))

    story.append(Paragraph(
        "This project is licensed under the <b>MIT License</b>. See the LICENSE file in the "
        "repository for details.",
        styles['CustomBody']
    ))

    # ============================================
    # CONTACT
    # ============================================

    story.append(Paragraph("8. Contact", styles['SectionHeader']))

    story.append(Paragraph(
        "For any questions or feedback, feel free to reach out:",
        styles['CustomBody']
    ))

    contact_info = [
        "<b>Email:</b> your.email@example.com",
        "<b>Twitter:</b> @yourhandle"
    ]

    for info in contact_info:
        story.append(Paragraph(
            f'<font color="#0D9488">&#x2022;</font>  {info}',
            styles['FeatureItem']
        ))

    story.append(Spacer(1, 30))

    # ============================================
    # FOOTER
    # ============================================

    story.append(Paragraph(
        "Thank you for your interest in the RAG-LLM Using AI Pipeline with Streamlit Interface!",
        styles['Footer']
    ))

    story.append(Spacer(1, 10))

    story.append(Paragraph(
        '<font color="#1E40AF">GitHub Repository: '
        'https://github.com/Dono1901/RAG-LLM-using-AI-Pipeline-with-streamlit-interface</font>',
        styles['Footer']
    ))

    # Build the PDF
    doc.build(story)
    print(f"PDF generated successfully: {output_path}")

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), "README.pdf")
    build_pdf(output_file)
