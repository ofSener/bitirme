#!/usr/bin/env python3
"""
Create PDF from LaTeX using online service
"""

import requests
import base64
import time

def create_pdf_from_latex(tex_file_path, output_pdf_path):
    """Convert LaTeX to PDF using online service"""

    # Read LaTeX file
    with open(tex_file_path, 'r') as f:
        latex_content = f.read()

    print("Converting LaTeX to PDF using online service...")
    print("This may take a few moments...")

    # Try LaTeX online compiler service
    url = "https://latexonline.cc/compile"

    # Prepare the request
    files = {
        'file': ('thesis.tex', latex_content, 'text/plain')
    }

    try:
        # Send request
        response = requests.post(
            url,
            files=files,
            params={'command': 'pdflatex'},
            timeout=60
        )

        if response.status_code == 200:
            # Save PDF
            with open(output_pdf_path, 'wb') as f:
                f.write(response.content)
            print(f"PDF successfully created: {output_pdf_path}")
            return True
        else:
            print(f"Error: Server returned status code {response.status_code}")
            return False

    except Exception as e:
        print(f"Error creating PDF: {e}")
        print("\nAlternative: You can use these online services:")
        print("1. https://www.overleaf.com - Upload the .tex file")
        print("2. https://latexbase.com - Paste the LaTeX code")
        print("3. https://latex.informatik.uni-halle.de/latex-online/latex.php")
        return False

if __name__ == "__main__":
    tex_file = "/Users/ofs/stajj/bitirme/kolmogorov_thesis.tex"
    pdf_file = "/Users/ofs/stajj/bitirme/kolmogorov_thesis.pdf"

    create_pdf_from_latex(tex_file, pdf_file)