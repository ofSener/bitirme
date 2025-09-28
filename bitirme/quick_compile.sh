#!/bin/bash

# Quick LaTeX Installation and Compilation Script
# For macOS users

echo "LaTeX Quick Setup for Graduation Project"
echo "========================================"
echo ""

# Check if LaTeX is installed
if command -v pdflatex &> /dev/null; then
    echo "✓ LaTeX is already installed!"
    echo "Compiling your thesis..."

    pdflatex -interaction=nonstopmode kolmogorov_thesis_final.tex
    pdflatex -interaction=nonstopmode kolmogorov_thesis_final.tex

    if [ -f "kolmogorov_thesis_final.pdf" ]; then
        echo ""
        echo "✓ SUCCESS! Your PDF has been created: kolmogorov_thesis_final.pdf"
        open kolmogorov_thesis_final.pdf
    else
        echo "⚠ Compilation failed. Check the .log file for errors."
    fi
else
    echo "LaTeX is not installed."
    echo ""
    echo "Choose an option:"
    echo "1) Install BasicTeX (90MB - recommended for quick use)"
    echo "2) Install MacTeX (4GB - full package)"
    echo "3) Use online compiler (open Overleaf)"
    echo ""
    read -p "Enter your choice (1/2/3): " choice

    case $choice in
        1)
            echo "Installing BasicTeX..."
            brew install --cask basictex
            echo ""
            echo "✓ Installation complete!"
            echo "Please restart your terminal and run this script again."
            ;;
        2)
            echo "Installing MacTeX (this will take a while)..."
            brew install --cask mactex
            echo ""
            echo "✓ Installation complete!"
            echo "Please restart your terminal and run this script again."
            ;;
        3)
            echo "Opening Overleaf..."
            open "https://www.overleaf.com"
            echo ""
            echo "Instructions:"
            echo "1. Create a free account"
            echo "2. Click 'New Project' → 'Upload Project'"
            echo "3. Upload kolmogorov_thesis_final.tex"
            echo "4. It will compile automatically"
            ;;
        *)
            echo "Invalid choice. Exiting."
            ;;
    esac
fi